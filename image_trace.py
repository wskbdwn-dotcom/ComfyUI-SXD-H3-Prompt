# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded stream validation for native image nodes; no third-party patching."""
from collections import deque
import re
from .live import LiveTrace

VERSION = '1.1.0'


class GuardError(RuntimeError):
    pass


def _cancel():
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return
    throw_exception_if_processing_interrupted()


def _close(iterator):
    close = getattr(iterator, 'close', None)
    if close:
        close()


class Repetition:
    """Four exact normalized long lines/groups, or repeated text without newlines."""
    def __init__(self):
        self.lines = deque(maxlen=48)
        self.partial = ''
        self.tail = ''
        self.count = 0

    def check_tail(self):
        for width in range(50, min(1024, len(self.tail) // 4) + 1):
            if self.tail[-width * 4:] == self.tail[-width:] * 4:
                raise GuardError('THINKING_LOOP: 思考区连续四遍重复长文本，已停止，未提交正文。')

    def feed(self, text):
        for line in (self.partial + text).split('\n')[:-1]:
            line = ' '.join(line.split())
            if not line:
                continue
            self.lines.append(line)
            recent = list(self.lines)
            for width in range(1, len(recent) // 4 + 1):
                block = recent[-width:]
                if sum(map(len, block)) * 4 >= 200 and recent[-width * 4:] == block * 4:
                    raise GuardError('THINKING_LOOP: 思考区连续四遍重复行组，已停止，未提交正文。')
        self.partial = (self.partial + text).rsplit('\n', 1)[-1][-16000:]
        for char in text:
            char = ' ' if char.isspace() else char
            if char == ' ' and self.tail.endswith(' '):
                continue
            self.tail = (self.tail + char)[-4096:]
            self.count += 1
            if self.count % 64 == 0:
                self.check_tail()


class Trace:
    def __init__(self, config, cancel=_cancel, validate=None, node_id=None):
        self.config, self.cancel = config, cancel
        self.validate = validate or (lambda text: text)
        self.raw, self.content, self.reasoning = [], [], []
        self.usage, self.finishes = {}, []
        self.characters = 0
        self.pending, self.thought_end = '', None
        self.saw_native_thought = False
        self.scan_done = False
        self.detector, self.reason_detector = Repetition(), Repetition()
        self.preview = LiveTrace(node_id)
        self.preview_thinking = False

    @property
    def raw_output(self):
        if self.raw:
            return ''.join(self.raw)
        thought, content = ''.join(self.reasoning), ''.join(self.content)
        return f'<think>{thought}</think>{content}' if thought else content

    def send(self, status):
        self.preview.stop = getattr(self, 'stop', [])
        self.preview.send(status, force=True)

    def _scan(self, text):
        if self.scan_done:
            return
        self.pending += text
        opens = {'<|channel>thought': '<channel|>', '<think>': '</think>'}
        while self.pending:
            if self.thought_end:
                end = self.pending.find(self.thought_end)
                if end >= 0:
                    if self.config.get('stop_on_repetition', True):
                        self.detector.feed(self.pending[:end])
                        self.detector.check_tail()
                    self.pending = self.pending[end + len(self.thought_end):]
                    self.thought_end = None
                    self.scan_done = True
                    self.pending = ''
                    return
                else:
                    keep = len(self.thought_end) - 1
                    if len(self.pending) <= keep:
                        return
                    if self.config.get('stop_on_repetition', True):
                        self.detector.feed(self.pending[:-keep])
                    self.pending = self.pending[-keep:]
                    return
            else:
                prefix = self.pending.lstrip()
                if prefix.startswith(': '):
                    prefix = prefix[2:].lstrip()
                match = next((tag for tag in opens if prefix.startswith(tag)), None)
                if match:
                    self.pending = prefix[len(match):]
                    self.thought_end = opens[match]
                    self.saw_native_thought = True
                elif not prefix or any(tag.startswith(prefix) for tag in opens) or prefix == ':':
                    return
                else:
                    # A final JSON string may literally mention a thinking marker.
                    self.scan_done = True
                    self.pending = ''
                    return

    def record(self, chunk, raw):
        self.cancel()
        usage = chunk.get('usage')
        if isinstance(usage, dict):
            self.usage.update(usage)
        for choice in chunk.get('choices', []):
            finish = choice.get('finish_reason')
            if finish is not None:
                self.finishes.append(finish)
            if raw:
                text = choice.get('text') or ''
                if text:
                    self.raw.append(text)
                self.preview.capture({'choices': [choice]})
                self._scan(text)
            else:
                delta = choice.get('delta') or {}
                text = delta.get('content') or ''
                self.content.append(text)
                thought = delta.get('reasoning_content') or ''
                self.reasoning.append(thought)
                # Some backends expose only parsed chat deltas. Never duplicate
                # these when the underlying completion stream was observed.
                if not self.raw:
                    visible = ''
                    if thought:
                        if not self.preview_thinking:
                            visible += '<think>'
                            self.preview_thinking = True
                        visible += thought
                    if text:
                        if self.preview_thinking:
                            visible += '</think>'
                            self.preview_thinking = False
                            self.preview.saw_thought_end = True
                        visible += text
                    self.preview.capture({'choices': [{'text': visible, 'finish_reason': finish}]})
                # Native completion has already been checked when it carries markers.
                if thought and not self.saw_native_thought and self.config.get('stop_on_repetition', True):
                    self.reason_detector.feed(thought)
                text += thought
            self.characters += len(text)
            if self.characters > 2_000_000:
                raise GuardError('OUTPUT_TOO_LARGE: 输出超出保护上限，未提交正文。')

    def observe(self, chunks):
        try:
            for chunk in chunks:
                self.record(chunk, True)
                yield chunk
        finally:
            _close(chunks)

    def answer(self):
        if 'length' in self.finishes:
            raise GuardError('OUTPUT_TRUNCATED: finish_reason=length，思考或正文被截断，未提交正文。')
        if not self.finishes or any(value != 'stop' for value in self.finishes):
            raise GuardError('INCOMPLETE_RESPONSE: 缺少正常结束状态，未提交正文。')
        if self.thought_end:
            raise GuardError('UNFINISHED_THINKING: 思考未正常结束，未提交正文。')
        if self.config.get('stop_on_repetition', True):
            self.reason_detector.check_tail()
        # Prefer parsed content when reasoning was separated by the backend.
        raw = ''.join(self.raw).strip()
        content = ''.join(self.content).strip()
        text = raw if raw and (self.saw_native_thought or not any(self.reasoning)) else content
        if text.startswith(': '):
            text = text[2:].strip()
        for opening, closing in (('<|channel>thought', '<channel|>'), ('<think>', '</think>')):
            while text.startswith(opening):
                end = text.find(closing, len(opening))
                if end < 0:
                    raise GuardError('UNFINISHED_THINKING: 仅返回了思考内容。')
                text = text[end + len(closing):].strip()
        if text.startswith('<|channel>final'):
            text = text[len('<|channel>final'):].strip()
        if not text:
            raise GuardError('EMPTY_ANSWER: 未返回最终正文。')
        # Keep normal JSON parsing and schema validation in the existing workflow.
        return text

    def diagnostic(self):
        tail = ''.join(self.raw or self.reasoning or self.content)[-2000:]
        counts = {k: v for k, v in self.usage.items() if type(v) is int}
        return (f'\n[Gemma guard {VERSION}] max_tokens={self.config.get("max_tokens")}, '
                f'n_ctx={self.config.get("n_ctx")}, usage={counts or "未提供"}'
                '\n已收到的原始输出末尾（仅供诊断，不是最终答案）：\n' + tail)

    def collect(self, chunks):
        if isinstance(chunks, dict):
            raise GuardError('BACKEND_INCOMPATIBLE: 后端未返回请求的流式结果。')
        try:
            for chunk in chunks:
                self.record(chunk, False)
        finally:
            _close(chunks)
        # Validate before runtime marks the call successful or keeps the model.
        answer = self.validate(self.answer())
        self.preview.saw_thought_end |= self.saw_native_thought and self.thought_end is None
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': answer}}],
                'usage': dict(self.usage)}

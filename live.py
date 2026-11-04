# SPDX-License-Identifier: GPL-3.0-or-later
"""Observe raw completion chunks without changing model sampling or stop tokens."""
import logging
import time
import uuid

_LOG = logging.getLogger(__name__)
EVENT = 'sxd-h3-stream'


class LiveTrace:
    def __init__(self, node_id=None):
        self.parts, self.pending = [], []
        self.usage, self.finish_reason = None, None
        self.stop, self.saw_thought_end = [], False
        self.tail = ''
        self.run = uuid.uuid4().hex
        self.node_id = str(node_id)
        self.sender, self.client = None, None
        self.last_send = 0.0
        try:
            from server import PromptServer
            server = PromptServer.instance
            if node_id is not None and server.client_id:
                self.sender, self.client = server.send_sync, server.client_id
        except (ImportError, AttributeError):
            pass  # Headless/API execution still returns the complete trace.

    @property
    def text(self):
        return ''.join(self.parts)

    def send(self, status, force=False):
        now = time.monotonic()
        if not force and now - self.last_send < 0.15:
            return
        delta = ''.join(self.pending)
        self.pending.clear()
        self.last_send = now
        if self.sender:
            try:
                self.sender(EVENT, {'node': self.node_id, 'run': self.run,
                    'delta': delta, 'status': status, 'finish_reason': self.finish_reason,
                    'thought_end_seen': self.saw_thought_end, 'stop': self.stop}, self.client)
            except Exception:
                _LOG.warning('H3 realtime preview disconnected; inference continues', exc_info=True)
                self.sender = None

    def capture(self, chunk):
        if chunk.get('usage'):
            self.usage = chunk['usage']
        for choice in chunk.get('choices', []):
            text = choice.get('text') or ''
            if text:
                self.parts.append(text)
                self.pending.append(text)
                scan = self.tail + text
                self.saw_thought_end |= '<channel|>' in scan
                self.tail = scan[-32:]
            if choice.get('finish_reason') is not None:
                self.finish_reason = choice['finish_reason']
        self.send('generating')

    def observe(self, chunks):
        try:
            for chunk in chunks:
                self.capture(chunk)
                yield chunk
        finally:
            close = getattr(chunks, 'close', None)
            if close:
                close()

    def collect(self, chunks):
        content, reasoning = [], []
        try:
            for chunk in chunks:
                if chunk.get('usage'):
                    self.usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta', {})
                    content.append(delta.get('content') or '')
                    reasoning.append(delta.get('reasoning_content') or '')
                    if choice.get('finish_reason') is not None:
                        self.finish_reason = choice['finish_reason']
        finally:
            close = getattr(chunks, 'close', None)
            if close:
                close()
        if self.finish_reason is None:
            raise RuntimeError('流式扩写未返回结束状态，未将不完整正文传给 H3。')
        # Native chunks preserve the thought markers and any text in the final
        # chunk (some chat converters omit that final chunk's text).
        return {'choices': [{'finish_reason': self.finish_reason, 'message': {
            'content': self.text if self.parts else ''.join(content),
            'reasoning_content': ''.join(reasoning)}}], 'usage': self.usage}


class H3PromptLiveDisplay:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'raw_output': ('STRING', {'forceInput': True})}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('raw_output',)
    FUNCTION = 'show'
    CATEGORY = 'H3/Author Gemma4'
    OUTPUT_NODE = True
    DESCRIPTION = '直接连接扩写节点第4个 raw_output 输出，实时查看原始思考和正文。报错时保留已收到的预览。不要将此输出接到 H3 提示词。'

    def show(self, raw_output):
        return {'ui': {'text': [raw_output]}, 'result': (raw_output,)}

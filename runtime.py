# SPDX-License-Identifier: GPL-3.0-or-later
"""Backend I2VA adapter for the Goohai optimizer rules and local Gemma4.

Author rules and postprocessing are vendored verbatim in author_rules.py.
No dependency on ComfyUI-llama-cpp_vlm, frontend hooks, or network services.
"""
from __future__ import annotations

import base64
import gc
import inspect
import io
import logging
import math
from pathlib import Path
import re
import threading

import folder_paths
from . import author_rules as author

_LOCK = threading.Lock()
_LOG = logging.getLogger(__name__)


def _roots():
    roots = [Path(folder_paths.models_dir) / name for name in ("LLM", "llm")]
    try:
        roots += [Path(p) for p in folder_paths.get_folder_paths("LLM")]
    except KeyError:
        pass
    return list(dict.fromkeys(p.resolve() for p in roots))


def _model_path(name):
    relative = Path(name)
    if not name or relative.is_absolute() or '..' in relative.parts or relative.suffix.lower() != '.gguf':
        raise ValueError("请输入 models/LLM 或 models/llm 下的 GGUF 相对文件名。")
    matches = []
    for root in _roots():
        path = (root / relative).resolve()
        # LLM and llm can name the same directory on Windows/macOS.
        if path.is_file() and not any(path.samefile(found) for found in matches):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"GGUF 文件需唯一匹配，实际找到 {len(matches)} 个：{name}")
    return matches[0]


def _backend():
    try:
        from llama_cpp import Llama, StoppingCriteriaList
        try:
            from llama_cpp.llama_multimodal import Gemma4ChatHandler
        except ImportError:
            from llama_cpp.llama_chat_format import Gemma4ChatHandler
    except ImportError as error:
        raise RuntimeError("需要支持 Gemma4ChatHandler 的 llama-cpp-python（JamePeng 构建）。无需安装旧扩写插件；请检查 ComfyUI 使用的 Python 环境。") from error
    # Never silently accept an ignored thinking toggle through **kwargs.
    if 'enable_thinking' not in inspect.signature(Gemma4ChatHandler.__init__).parameters:
        raise RuntimeError("当前 Gemma4ChatHandler 未明确支持 enable_thinking，请使用支持该参数的构建。")
    return Llama, Gemma4ChatHandler, StoppingCriteriaList


def _projector_keyword(handler_class):
    for cls in handler_class.__mro__:
        parameters = inspect.signature(cls.__init__).parameters
        for key in ('mmproj_path', 'clip_model_path'):
            if key in parameters:
                return key
    raise RuntimeError("Gemma4 处理器缺少可识别的 mmproj_path / clip_model_path 参数。")


def _image_data(first_frame):
    import numpy as np
    from PIL import Image
    if len(first_frame.shape) != 4 or first_frame.shape[0] < 1 or first_frame.shape[-1] not in (3, 4):
        raise ValueError("首帧必须是 ComfyUI 的非空 IMAGE 张量 [B,H,W,C]。")
    pixels = np.clip(first_frame[0].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    picture = Image.fromarray(pixels).convert('RGB')
    # Same longest-edge cap and JPEG quality as the author's browser input.
    # Browser canvas and Pillow resampling are not byte-identical encoders.
    scale = min(1.0, 512.0 / max(picture.size))
    size = tuple(max(1, math.floor(dimension * scale + 0.5)) for dimension in picture.size)
    if size != picture.size:
        picture = picture.resize(size, Image.Resampling.BICUBIC)
    stream = io.BytesIO()
    picture.save(stream, format='JPEG', quality=42)
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def _messages(prompt, image_data, duration, output_language):
    label = '<picture 1>'
    context = {
        'main_mode': 'text_keyframes',
        'keyframes': [{'label': label, 'role': 'exact_first_frame'}],
        'audio_mode': 'native',
    }
    system = author._system_prompt('I2VA', duration, [label], output_language, context, prompt)
    # Matches the author's _gguf_generate message order and image schema.
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': [
            {'type': 'text', 'text': 'User prompt:\n' + prompt},
            {'type': 'text', 'text': label + ' visual 1.'},
            {'type': 'image_url', 'image_url': {'url': image_data}},
        ]},
    ]


def _final_content(response, *, max_tokens=None, n_ctx=None, enable_thinking=None):
    choice = response['choices'][0]
    if choice.get('finish_reason') == 'length':
        usage = response.get('usage') or {}
        if not isinstance(usage, dict):
            usage = {}
        def count(name):
            value = usage.get(name)
            return value if type(value) is int and value >= 0 else '未提供'
        raise RuntimeError(
            "扩写被截断（finish_reason=length），未将不完整正文传给 H3。"
            f" 请求参数：max_tokens={max_tokens}, n_ctx={n_ctx}, enable_thinking={enable_thinking}。"
            f" 后端用量：prompt_tokens={count('prompt_tokens')}, "
            f"completion_tokens={count('completion_tokens')}, total_tokens={count('total_tokens')}。"
            " 思考与最终正文共享生成额度；输入规则和图片也占用上下文。"
            " 请保持所需的思考开关，检查输出预算和剩余上下文；"
            "仅凭 length 不能区分两者，也不能排除模型重复生成。"
        )
    # Do not merge reasoning_content into the H3 prompt. Some Gemma backends
    # instead return a leading native thought channel inside content.
    text = choice['message'].get('content')
    if not isinstance(text, str):
        raise RuntimeError("Gemma4 未返回最终正文，请检查思考开关、token 上限和模型兼容性。")
    text = text.removeprefix(': ').strip()
    for opening, closing in [('<think>', '</think>'), ('<|channel>thought', '<channel|>')]:
        while text.startswith(opening):
            end = text.find(closing, len(opening))
            if end < 0:
                raise RuntimeError("Gemma4 仅返回了未完成的思考内容，没有最终扩写正文。")
            text = text[end + len(closing):].strip()
    text = text.removeprefix('<|channel>final').strip()
    if not text:
        raise RuntimeError("Gemma4 返回了空正文。")
    return text


def _postprocess(text, prompt, duration, output_language, task='I2VA'):
    # The same call order and arguments as upstream _run_prompt_optimization.
    formatted = author._format_prompt_sections(author._strip_filenames(text))
    formatted = author._ensure_fl2va_picture_labels(formatted, task, duration, output_language)
    formatted = author._ensure_supplied_dialogues(formatted, prompt, output_language)
    return author._normalize_subject_shorthand(formatted)


def _chat_with_interrupt(llm, criteria, **kwargs):
    """Chat forwards to create_completion, where stopping_criteria is supported.

    Bind cancellation only to this adapter-owned instance and restore it even
    on failure. Never patch the Llama class or third-party source files.
    """
    original = llm.create_completion
    had_override = "create_completion" in vars(llm)

    def complete(*args, **options):
        existing = options.get('stopping_criteria')
        options['stopping_criteria'] = type(criteria)(
            [*criteria, *(existing or [])])
        return original(*args, **options)

    try:
        llm.create_completion = complete
        return llm.create_chat_completion(**kwargs)
    finally:
        if had_override:
            llm.create_completion = original
        else:
            del llm.create_completion


# One resident pair per plugin module, independent of ComfyUI's node-output LRU.
# All access (including release) is serialized by _LOCK.
_RESIDENT = None


def _file_identity(path):
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _close_pair(llm, handler):
    # The projector may retain a reference to the Llama model/context.
    for resource in (handler, llm):
        if resource is None:
            continue
        try:
            close = getattr(resource, 'close', None)
            if callable(close):
                close()
            elif getattr(resource, '_exit_stack', None) is not None:
                resource._exit_stack.close()
        except Exception:
            _LOG.exception('Failed to release a Gemma4 inference resource')


def _release_resident():
    """Caller holds _LOCK. Never offload models owned by ComfyUI."""
    global _RESIDENT
    pair, _RESIDENT = _RESIDENT, None
    if pair is not None:
        _, llm, handler = pair
        _close_pair(llm, handler)
        _LOG.info('H3 Gemma4: RELEASE resident model and projector')
        return True
    return False


def _reset_inference_state(llm):
    """Reset one request, retaining loaded weights and the vision projector.

    JamePeng 0.3.35 Llama.reset() only sets n_tokens=0. Its MTMD path can
    then eval at position zero over stale native KV/SWA state. Explicitly
    clear native memory and invalidate hybrid checkpoints as well.
    """
    llm.reset()
    clear = getattr(getattr(llm, '_ctx', None), 'memory_clear', None)
    if not callable(clear):
        raise RuntimeError('当前 Llama 缺少底层 memory_clear，无法安全复用模型；已停止扩写。')
    clear(True)
    checkpoints = getattr(llm, '_hybrid_cache_mgr', None)
    if checkpoints is not None:
        clear_checkpoints = getattr(checkpoints, 'clear', None)
        if not callable(clear_checkpoints):
            raise RuntimeError('当前 Llama 无法清理混合模型检查点，无法安全复用模型。')
        clear_checkpoints()


def _resident_model(Llama, Handler, model_path, projector_path,
                    enable_thinking, n_ctx, n_gpu_layers, mm):
    global _RESIDENT
    key = (_file_identity(model_path), _file_identity(projector_path),
           bool(enable_thinking), int(n_ctx), int(n_gpu_layers))
    if _RESIDENT is not None and _RESIDENT[0] == key:
        _LOG.info('H3 Gemma4: REUSE resident model and projector')
        return _RESIDENT[1]
    _release_resident()
    gc.collect()
    mm.soft_empty_cache()
    llm = handler = None
    try:
        handler = Handler(**{
            _projector_keyword(Handler): str(projector_path),
            'enable_thinking': bool(enable_thinking), 'verbose': False,
        })
        llm = Llama(model_path=str(model_path), chat_handler=handler,
                    n_gpu_layers=int(n_gpu_layers), n_ctx=int(n_ctx), verbose=False)
        if not callable(getattr(llm, 'reset', None)):
            raise RuntimeError('当前 Llama 不支持 reset，无法安全复用；请检查本地依赖版本。')
    except BaseException:
        _close_pair(llm, handler)
        raise
    _RESIDENT = (key, llm, handler)
    _LOG.info('H3 Gemma4: LOAD resident model and projector')
    return llm


class H3AuthorGemma4Release:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'release': ('BOOLEAN', {'default': True})}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('status',)
    FUNCTION = 'release'
    CATEGORY = 'H3/Author Gemma4'
    OUTPUT_NODE = True
    DESCRIPTION = '手动释放本版本扩写节点常驻的 Gemma 和投影器，不卸载 H3。请单独运行，不放入常用扩写流程。'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def release(self, release=True):
        if not release:
            return ('未释放：release 已关闭。',)
        from comfy import model_management as mm
        with _LOCK:
            released = _release_resident()
            gc.collect()
            mm.soft_empty_cache()
        return ('已释放 Gemma 和投影器，下一次扩写会重新加载。' if released
                else '本版本没有已缓存的 Gemma。',)


class H3AuthorGemma4Expand:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'prompt': ('STRING', {'forceInput': True}),
            'first_frame': ('IMAGE',),
            'frame_count': ('INT', {'forceInput': True}),
            'seed': ('INT', {'forceInput': True}),
            'model': ('STRING', {'default': 'gemma-4-26b-a4b-it-heretic.q4_k_m.gguf'}),
            'mmproj': ('STRING', {'default': 'gemma-4-26B-A4B-it-heretic-mmproj.bf16.gguf'}),
            'enable_thinking': ('BOOLEAN', {'default': False, 'tooltip': '直接传给 Gemma4 处理器，无需修改插件源码。26B-A4B / 31B 可切换。'}),
            'output_language': (['中文', 'English'], {'default': '中文'}),
            'max_tokens': ('INT', {'default': 4096, 'min': 512, 'max': 1048576,
                'tooltip': '思考与最终正文共享的输出额度。界面上限不是模型容量保证；实际受 n_ctx 剩余空间及后端限制。'}),
            'n_ctx': ('INT', {'default': 16384, 'min': 2048, 'max': 1048576, 'step': 128,
                'tooltip': '输入、图片、思考与正文共用的上下文容量。请按模型支持范围和显存设置；界面允许更大数值不代表模型支持。修改后会重新加载模型。'}),
            'n_gpu_layers': ('INT', {'default': -1, 'min': -1, 'max': 999, 'tooltip': '-1 全部可卸载层放 GPU；0 使用 CPU。'}),
            'fps': ('FLOAT', {'default': 24.0, 'min': 1.0, 'max': 120.0}),
        }, 'optional': {
            'release_after_expansion': ('BOOLEAN', {'default': False,
                'label_on': '释放', 'label_off': '保留',
                'tooltip': '扩写后释放 Gemma：关闭则保留模型和投影器供下次复用；开启则本轮扩写结束后释放。只影响 Gemma，不卸载 H3。旧 API 不传此项时默认保留。'}),
        }}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('optimized_prompt',)
    FUNCTION = 'expand'
    CATEGORY = 'H3/Author Gemma4'
    DESCRIPTION = '保留 Goohai 作者扩写规则与后处理，使用本地 Gemma4。思考开关直接生效；无旧扩写节点、二次审核或静默回退。'

    @classmethod
    def IS_CHANGED(cls, model, mmproj, **kwargs):
        files = [_model_path(name).stat() for name in (model, mmproj)]
        return repr([(item.st_size, item.st_mtime_ns) for item in files])

    def expand(self, prompt, first_frame, frame_count, seed, model, mmproj,
               enable_thinking=False, output_language='中文', max_tokens=4096,
               n_ctx=16384, n_gpu_layers=-1, fps=24.0,
               _messages_override=None, _task="I2VA", _author_postprocess=True,
               release_after_expansion=False):
        from comfy import model_management as mm
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise ValueError("视频帧数及帧率必须为正数。")
        if re.search(r'(?:^|[-_. ])e[24]b(?:[-_. ]|$)', model.lower()) and not enable_thinking:
            raise ValueError("此适配默认面向 Gemma4 26B-A4B / 31B；E2B/E4B 构建不支持关闭思考，请开启 enable_thinking。")
        model_path, projector_path = _model_path(model), _model_path(mmproj)
        Llama, Handler, StoppingCriteriaList = _backend()
        duration = frame_count / fps
        messages = (_messages(prompt, _image_data(first_frame), duration, output_language)
                    if _messages_override is None else _messages_override)

        def interrupt(*args):
            if mm.processing_interrupted():
                raise mm.InterruptProcessingException()
            return False

        # Only model weights/projector/context allocations persist. Each request
        # receives fresh messages and resets inference state, not the weights.
        with _LOCK:
            interrupt()
            try:
                llm = _resident_model(Llama, Handler, model_path, projector_path,
                                      enable_thinking, n_ctx, n_gpu_layers, mm)
                interrupt()
                _reset_inference_state(llm)
                _LOG.info('H3 Gemma4: GENERATE thinking=%s max_tokens=%d n_ctx=%d',
                          bool(enable_thinking), int(max_tokens), int(n_ctx))
                response = _chat_with_interrupt(
                    llm, StoppingCriteriaList([interrupt]),
                    messages=messages, seed=int(seed) % (2**32),
                    max_tokens=int(max_tokens), temperature=0.2, top_p=0.9,
                )
                interrupt()
                text = _final_content(response, max_tokens=int(max_tokens),
                                      n_ctx=int(n_ctx), enable_thinking=bool(enable_thinking))
                result = (_postprocess(text, prompt, duration, output_language, _task)
                          if _author_postprocess else text)
                if not result.strip():
                    raise RuntimeError("作者后处理得到空提示词，任务已停止。")
                if release_after_expansion:
                    _release_resident()
                    gc.collect()
                    mm.soft_empty_cache()
                else:
                    _reset_inference_state(llm)
                return (result,)
            except BaseException:
                # Cancelled/failed native inference must not poison the next run.
                # A healthy resident model is kept on ordinary successful runs.
                _release_resident()
                gc.collect()
                mm.soft_empty_cache()
                raise

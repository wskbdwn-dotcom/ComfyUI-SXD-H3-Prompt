# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenAI-compatible text/image calls with an exact original-prompt fallback."""

import base64
import io
import json
import http.client
import re
from urllib import error, parse, request


class _ImageInputs(dict):
    """Expose one socket while validating additional numbered IMAGE inputs."""

    def __contains__(self, key):
        return super().__contains__(key) or bool(re.fullmatch(r'image_[1-9][0-9]*', str(key)))

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self:
            return ('IMAGE',)
        raise KeyError(key)


def _user_content(prompt, image_inputs):
    # Preserve the existing text-only request exactly; import image libraries lazily.
    batches = [(int(name[6:]), batch) for name, batch in image_inputs.items()
               if re.fullmatch(r'image_[1-9][0-9]*', name) and batch is not None]
    if not batches:
        return prompt
    import numpy as np
    from PIL import Image

    content = [{'type': 'text', 'text': prompt}]
    image_number = 0
    for slot, batch in sorted(batches):
        if getattr(batch, 'ndim', None) != 4 or batch.shape[-1] not in (1, 3, 4):
            raise ValueError('invalid_images')
        for frame_number, frame in enumerate(batch, 1):
            if hasattr(frame, 'detach'):
                frame = frame.detach().cpu().float().numpy()
            array = np.asarray(frame)
            if not np.isfinite(array).all() or not array.shape[0] or not array.shape[1]:
                raise ValueError('invalid_images')
            array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
            if array.shape[-1] == 1:
                array = array[:, :, 0]
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            image_number += 1
            content.extend([
                {'type': 'text', 'text': f'<Image {image_number}> (image_{slot}, item {frame_number})'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}},
            ])
    return content if image_number else prompt


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise error.HTTPError(req.full_url, code, msg, headers, fp)


def _open(req, timeout):
    return request.build_opener(_NoRedirect).open(req, timeout=timeout)


def _endpoint(base_url):
    if not isinstance(base_url, str):
        raise ValueError('invalid_url')
    url = base_url.strip().rstrip('/')
    parts = parse.urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.netloc or parts.query or parts.fragment:
        raise ValueError('invalid_url')
    if not parts.path.endswith('/chat/completions'):
        url += '/chat/completions'
    return url


def _json_object(value, label):
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError('invalid_' + label)
    return parsed


def _headers(api_key, extra_headers_json):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'ComfyUI-SXD-H3-Prompt/1.0',
    }
    extra = _json_object(extra_headers_json, 'headers')
    for name, value in extra.items():
        if not isinstance(name, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
            raise ValueError('invalid_headers')
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        if not isinstance(value, str) or '\r' in value or '\n' in value:
            raise ValueError('invalid_headers')
        headers[name] = value
    if isinstance(api_key, str) and api_key.strip() and not any(k.lower() == 'authorization' for k in headers):
        headers['Authorization'] = 'Bearer ' + api_key.strip()
    return headers


def _content(data):
    if not isinstance(data, dict) or data.get('error'):
        raise ValueError('invalid_response')
    choices = data.get('choices')
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError('invalid_response')
    choice = choices[0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('incomplete_response')
    message = choice.get('message')
    if not isinstance(message, dict):
        raise ValueError('invalid_response')
    content = message.get('content')
    if isinstance(content, list):
        if not content or not all(isinstance(item, dict) and item.get('type') == 'text'
                                  and isinstance(item.get('text'), str) for item in content):
            raise ValueError('invalid_response')
        content = ''.join(item['text'] for item in content)
    if not isinstance(content, str) or not content.strip():
        raise ValueError('empty_response')
    return content


def _fallback_reason(exc):
    # Never include provider error bodies, URLs, headers, or keys in a workflow output.
    if isinstance(exc, error.HTTPError):
        return 'http_' + str(exc.code)
    if isinstance(exc, error.URLError):
        if isinstance(exc.reason, TimeoutError):
            return 'timeout'
        return 'network_error'
    if isinstance(exc, TimeoutError):
        return 'timeout'
    if isinstance(exc, json.JSONDecodeError):
        return 'invalid_json'
    if isinstance(exc, ValueError) and str(exc) in {
            'invalid_url', 'invalid_headers', 'invalid_body', 'invalid_response',
            'incomplete_response', 'empty_response', 'invalid_model', 'invalid_images'}:
        return str(exc)
    return 'request_error'


class SXDOpenAITextFallback:
    """Use Chat Completions and keep the original prompt on any API failure."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'original_prompt': ('STRING', {'default': '', 'multiline': True}),
            'system_prompt': ('STRING', {'default': '', 'multiline': True}),
            'api_base_url': ('STRING', {'default': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'}),
            'api_key': ('STRING', {'default': ''}),
            'model': ('STRING', {'default': 'qwen-plus'}),
            'temperature': ('FLOAT', {'default': 0.2, 'min': 0.0, 'max': 2.0, 'step': 0.05}),
            'top_p': ('FLOAT', {'default': 1.0, 'min': 0.01, 'max': 1.0, 'step': 0.01}),
            'max_tokens': ('INT', {'default': 1024, 'min': 1, 'max': 65536}),
            'max_tokens_field': (['max_tokens', 'max_completion_tokens'], {'default': 'max_tokens'}),
            'timeout_sec': ('INT', {'default': 60, 'min': 1, 'max': 3600}),
            'extra_headers_json': ('STRING', {'default': '{}', 'multiline': True}),
            'extra_body_json': ('STRING', {'default': '{}', 'multiline': True}),
        }, 'optional': _ImageInputs({
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0x1FFFFFFFFFFFFF,
                             'tooltip': '仅控制 ComfyUI 缓存；改变后重新请求 API，不传给模型。'}),
            'image_1': ('IMAGE', {'tooltip': '可选图片或图片批次；连线后自动增加下一图片接口。'}),
        })}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('text', 'status')
    FUNCTION = 'run'
    CATEGORY = 'SXD/LLM'
    DESCRIPTION = 'OpenAI 兼容文本/多图调用；图片按接口编号及批次顺序发送。失败时返回原文，status 显示原因。'

    def run(self, original_prompt, system_prompt, api_base_url, api_key, model,
            temperature=0.2, top_p=1.0, max_tokens=1024, max_tokens_field='max_tokens', timeout_sec=60,
            extra_headers_json='{}', extra_body_json='{}', seed=0, **image_inputs):
        # ComfyUI includes seed in its input signature; the remote API does not receive it.
        _ = seed
        try:
            endpoint = _endpoint(api_base_url)
            if not isinstance(model, str) or not model.strip():
                raise ValueError('invalid_model')
            if max_tokens_field not in ('max_tokens', 'max_completion_tokens'):
                raise ValueError('invalid_body')
            headers = _headers(api_key, extra_headers_json)
            messages = []
            if isinstance(system_prompt, str) and system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': _user_content(original_prompt, image_inputs)})
            body = {'model': model.strip(), 'messages': messages,
                    'temperature': temperature, 'top_p': top_p, max_tokens_field: max_tokens}
            extra = _json_object(extra_body_json, 'body')
            if any(key in body or key in ('max_tokens', 'max_completion_tokens') for key in extra) or extra.get('stream') is True:
                raise ValueError('invalid_body')
            body.update(extra)
            payload = json.dumps(body, ensure_ascii=False).encode('utf-8')
            req = request.Request(endpoint, data=payload, headers=headers, method='POST')
            with _open(req, timeout_sec) as response:
                data = json.loads(response.read().decode('utf-8'))
            return (_content(data), 'success')
        except (error.HTTPError, error.URLError, TimeoutError, OSError,
                ValueError, TypeError, UnicodeError, http.client.HTTPException) as exc:
            return (original_prompt, 'fallback: ' + _fallback_reason(exc))

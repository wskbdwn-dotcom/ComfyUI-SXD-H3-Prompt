# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenAI-compatible text expansion with an exact original-prompt fallback."""

import json
import http.client
import re
from urllib import error, parse, request


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
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
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
            'incomplete_response', 'empty_response', 'invalid_model'}:
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
        }}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('text', 'status')
    FUNCTION = 'run'
    CATEGORY = 'SXD/LLM'
    DESCRIPTION = 'OpenAI 兼容 Chat Completions；失败时 text 原样输出 original_prompt。status 显示成功或回退原因。'

    def run(self, original_prompt, system_prompt, api_base_url, api_key, model,
            temperature=0.2, top_p=1.0, max_tokens=1024, max_tokens_field='max_tokens', timeout_sec=60,
            extra_headers_json='{}', extra_body_json='{}'):
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
            messages.append({'role': 'user', 'content': original_prompt})
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

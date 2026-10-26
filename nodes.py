# SPDX-License-Identifier: GPL-3.0-or-later
"""Editable author rules and image routing; no changes to video conditioning."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from . import runtime

DEFAULTS = json.loads(Path(__file__).with_name('rule_templates.json').read_text())
KEYFRAMES = '首尾帧'
REFERENCES = '全能参考（多图）'
_TOKEN = re.compile(r'\{\{([^{}]+)\}\}')


def render_rules(template, task, duration, labels, context, language, prompt):
    """A small text-only renderer. It never evaluates Python or user expressions."""
    if not isinstance(template, str) or not template.strip():
        raise ValueError('当前模式的规则框为空，请填写规则或恢复默认值。')
    roles = []
    for item in context['keyframes']:
        role = ('is the exact first-frame anchor at 0.00s'
                if item['role'] == 'exact_first_frame'
                else f'is the exact last-frame anchor at {duration:.2f}s')
        roles.append(f"{item['label']} {role}")
    keyframes = ('Keyframe roles supplied by the node: ' + '; '.join(roles)
                 + '. Preserve these roles exactly.' if roles else
                 'No concrete keyframe role was supplied by the node; do not invent one.')
    values = {'task': task, 'duration': f'{duration:.2f}',
              'labels': ', '.join(labels) or 'none', 'keyframe_roles': keyframes,
              'user_prompt': prompt}
    flags = {'FL2VA': task == 'FL2VA', 'not_FL2VA': task != 'FL2VA',
             'zh': language == '中文', 'en': language != '中文',
             'singing': bool(re.search(DEFAULTS['singing_pattern'], prompt, re.I)),
             'transfer': bool(re.search(DEFAULTS['transfer_pattern'], prompt, re.I))}
    output, stack, active, previous = [], [], True, 0
    for match in _TOKEN.finditer(template):
        if active:
            output.append(template[previous:match.start()])
        token = match.group(1).strip()
        if token.startswith('#'):
            name = token[1:]
            if name not in flags:
                raise ValueError(f'未知规则条件：{token}')
            stack.append((name, active))
            active = active and flags[name]
        elif token.startswith('/'):
            if not stack or stack[-1][0] != token[1:]:
                raise ValueError(f'规则条件标签不匹配：{token}')
            _, active = stack.pop()
        else:
            if token not in values:
                raise ValueError(f'未知规则变量：{token}')
            if active:
                output.append(values[token])
        previous = match.end()
    if stack:
        raise ValueError(f'规则条件未闭合：{stack[-1][0]}')
    if active:
        output.append(template[previous:])
    result = ''.join(output)
    if not result.strip():
        raise ValueError('当前模式的规则展开后为空。')
    return result


def _one_image(image, name):
    if len(image.shape) != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError(f'{name} 需要单张 IMAGE；图片批次请接 reference_batch。')
    return image


def prepare(prompt, mode, keyframe_rules, reference_rules, frame_count, fps,
            language, first_frame=None, last_frame=None, reference_batch=None,
            **references):
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError('视频帧数和帧率必须为正数。')
    duration = frame_count / fps
    media, keyframes = [], []
    if mode == KEYFRAMES:
        task = ('FL2VA' if last_frame is not None else 'I2VA') if first_frame is not None else (
            'L2VA' if last_frame is not None else 'T2VA')
        for name, image, role in [('first_frame', first_frame, 'exact_first_frame'),
                                  ('last_frame', last_frame, 'exact_last_frame')]:
            if image is None:
                continue
            ordinal = len(media) + 1
            label = f'picture {ordinal}' if task == 'FL2VA' else f'<picture {ordinal}>'
            media.append((label, _one_image(image, name), name))
            keyframes.append({'label': label, 'role': role})
        template = keyframe_rules
        context = {'main_mode': 'text_keyframes', 'keyframes': keyframes, 'audio_mode': 'native'}
    elif mode == REFERENCES:
        task = 'Ref2VA'
        sources = [('first_frame', first_frame)] + [
            (f'reference_{i}', references.get(f'reference_{i}')) for i in range(2, 10)]
        for name, image in sources:
            if image is not None:
                media.append((f'<picture {len(media)+1}>', _one_image(image, name), name))
        if reference_batch is not None:
            if len(reference_batch.shape) != 4 or reference_batch.shape[0] < 1:
                raise ValueError('reference_batch 需要非空 IMAGE 批次。')
            for index in range(reference_batch.shape[0]):
                image = _one_image(reference_batch[index:index+1], 'reference_batch')
                media.append((f'<picture {len(media)+1}>', image, f'reference_batch[{index}]'))
        if not 1 <= len(media) <= 9:
            raise ValueError('全能参考（多图）模式需要 1–9 张图片；请检查重复连接或批次大小。')
        template = reference_rules
        context = {'main_mode': 'all_reference', 'keyframes': [], 'audio_mode': 'native'}
    else:
        raise ValueError(f'未知模式：{mode}')

    labels = [label for label, _, _ in media]
    system = render_rules(template, task, duration, labels, context, language, prompt)
    content = [{'type': 'text', 'text': 'User prompt:\n' + prompt}]
    for label, image, _ in media:
        # One visual per author media item, hence "visual 1" for each image.
        content += [{'type': 'text', 'text': label + ' visual 1.'},
                    {'type': 'image_url', 'image_url': {'url': runtime._image_data(image)}}]
    ignored = []
    if mode == REFERENCES and last_frame is not None:
        ignored.append('last_frame（全能参考模式不把尾帧自动当参考图）')
    if mode == KEYFRAMES:
        ignored += [name for name, value in references.items() if value is not None]
        if reference_batch is not None:
            ignored.append('reference_batch')
    mapping = json.dumps({'mode': mode, 'task': task, 'duration_seconds': duration,
                          'media': [{'label': label, 'input': name} for label, _, name in media],
                          'keyframes': keyframes, 'ignored_inputs': ignored,
                          'scope': '提示词扩写图片；不会改变视频生成节点的图片输入'},
                         ensure_ascii=False, indent=2)
    return task, [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}], mapping


class H3AuthorGemma4ExpandV2(runtime.H3AuthorGemma4Expand):
    @classmethod
    def INPUT_TYPES(cls):
        required = dict(super().INPUT_TYPES()['required'])
        required.pop('first_frame')
        required.update({
            'mode': ([KEYFRAMES, REFERENCES], {'default': KEYFRAMES}),
            'keyframe_rules': ('STRING', {'multiline': True, 'default': DEFAULTS['keyframe_rules'],
                'tooltip': '首尾帧规则全文。只有首帧为 I2VA，只有尾帧为 L2VA，两者都有为 FL2VA，均未接入为 T2VA。'}),
            'reference_rules': ('STRING', {'multiline': True, 'default': DEFAULTS['reference_rules'],
                'tooltip': '全能参考（仅图片）规则全文；默认作者六段式。切换模式不会覆盖另一规则框。'}),
            'author_postprocess': ('BOOLEAN', {'default': True,
                'tooltip': '默认保留作者格式整理、对白保留和首尾帧标签补全；自定义输出格式时可关闭。'}),
        })
        optional = {'first_frame': ('IMAGE', {'tooltip': '首帧；全能参考模式中作为第一张参考图。'}),
                    'last_frame': ('IMAGE', {'tooltip': '仅首尾帧模式使用；全能参考模式忽略。'})}
        optional.update({f'reference_{i}': ('IMAGE', {'tooltip': '仅全能参考模式使用，按已连接输入顺序连续编号。'})
                         for i in range(2, 10)})
        optional['reference_batch'] = ('IMAGE', {'tooltip': '仅全能参考模式使用；追加在独立参考图之后，总数最多9张。'})
        # Append the optional widget after every pre-existing widget.
        optional.update(super().INPUT_TYPES().get('optional', {}))
        return {'required': required, 'optional': optional}

    RETURN_TYPES = ('STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('optimized_prompt', 'applied_rules', 'media_mapping')
    CATEGORY = 'H3/Author Gemma4'
    DESCRIPTION = '两个可编辑规则框，按模式选择；只修改扩写，不给下游视频生成节点自动添加尾帧或参考图。'

    def expand(self, prompt, frame_count, seed, model, mmproj,
               enable_thinking=False, output_language='中文', max_tokens=4096,
               n_ctx=16384, n_gpu_layers=-1, fps=24.0, mode=KEYFRAMES,
               keyframe_rules=DEFAULTS['keyframe_rules'], reference_rules=DEFAULTS['reference_rules'],
               author_postprocess=True, first_frame=None, last_frame=None, reference_batch=None,
               release_after_expansion=False, **references):
        allowed = {f'reference_{i}' for i in range(2, 10)}
        if set(references) - allowed:
            raise ValueError('未知参考图输入：' + ', '.join(sorted(set(references) - allowed)))
        task, messages, mapping = prepare(prompt, mode, keyframe_rules, reference_rules,
            frame_count, fps, output_language, first_frame, last_frame, reference_batch, **references)
        result = super().expand(prompt, first_frame, frame_count, seed, model, mmproj,
            enable_thinking, output_language, max_tokens, n_ctx, n_gpu_layers, fps,
            _messages_override=messages, _task=task, _author_postprocess=author_postprocess,
            release_after_expansion=release_after_expansion)
        return result[0], messages[0]['content'], mapping

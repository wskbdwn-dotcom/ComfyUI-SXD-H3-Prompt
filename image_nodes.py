# SPDX-License-Identifier: GPL-3.0-or-later
"""Native image prompting/checking using the existing shared Gemma runtime."""
import base64
import io
import json
import math
from pathlib import Path

from . import runtime
from .image_trace import Trace
from . import spatial_plan

TYPES = {'extra_arm', 'extra_leg', 'extra_head', 'extra_torso', 'missing_arm',
         'missing_leg', 'body_twist', 'bad_connection'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'JSON 字段重复：{key}')
        result[key] = value
    return result


def _json(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_answer(text, task, user_request=''):
    """Normalize field order, never turn an inference/format error into pass."""
    if task == 'spatial_plan':
        return spatial_plan.normalized_plan(text, user_request)
    text = text.strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    value = _json(text)
    if not isinstance(value, dict):
        raise ValueError('最终正文必须是 JSON 对象。')
    if task == 'prompt':
        if set(value) != {'prompt'} or not isinstance(value['prompt'], str) or not value['prompt'].strip():
            raise ValueError('扩写必须输出非空字符串字段 prompt，不接受其他字段。')
        result = {'prompt': value['prompt'].strip()}
    elif task == 'anatomy':
        if set(value) != {'status', 'issues'} or value['status'] not in ('pass', 'uncertain', 'repair'):
            raise ValueError('检查必须输出 status 和 issues，状态只能是 pass / uncertain / repair。')
        issues = value['issues']
        if not isinstance(issues, list):
            raise ValueError('issues 必须是数组。')
        if (value['status'] == 'repair') != bool(issues):
            raise ValueError('repair 必须有问题；pass 和 uncertain 必须使用空 issues。')
        normalized = []
        for issue in issues:
            if not isinstance(issue, dict) or set(issue) != {'type', 'location', 'evidence', 'fix'}:
                raise ValueError('问题字段必须为 type、location、evidence、fix。')
            if not all(isinstance(v, str) and v.strip() for v in issue.values()):
                raise ValueError('每个问题字段都必须为非空字符串。')
            if issue['type'] not in TYPES:
                raise ValueError('不支持的问题类型：' + issue['type'])
            normalized.append({key: issue[key].strip() for key in ('type', 'location', 'evidence', 'fix')})
        result = {'status': value['status'], 'issues': normalized}
    else:
        raise ValueError('task 必须为 prompt、anatomy 或内部 spatial_plan。')
    return json.dumps(result, ensure_ascii=False, separators=(',', ':'))


def _resolve(name):
    if not isinstance(name, str) or not name:
        raise ValueError('配置缺少 model_path 或 mmproj_path。')
    path = Path(name)
    if not path.is_absolute():
        return runtime._model_path(name)
    if path.suffix.lower() != '.gguf' or not path.is_file():
        raise FileNotFoundError('GGUF 文件不存在或后缀错误：' + name)
    return path.resolve()


def parse_config(text):
    config = _json(text)
    if not isinstance(config, dict):
        raise ValueError('模型配置必须是 JSON 对象。')
    defaults = dict(enable_thinking=True, max_tokens=8192, n_ctx=32768,
                    n_gpu_layers=-1, temperature=1.0, top_k=64, top_p=.95, min_p=0.,
                    repeat_penalty=1.08, penalty_last_n=4096, stop_on_repetition=True,
                    max_images=4, image_max_side=1024, image_jpeg_quality=90)
    allowed = set(defaults) | {'model_path', 'mmproj_path'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError('原生节点不使用这些配置字段，请导入 v6 工作流：' + ', '.join(sorted(unknown)))
    defaults.update(config)
    for name in ('enable_thinking', 'stop_on_repetition'):
        if type(defaults[name]) is not bool:
            raise ValueError(name + ' 必须为布尔值。')
    for name, low, high in (('max_tokens', 1, 1048576), ('n_ctx', 2048, 1048576),
            ('n_gpu_layers', -1, 999), ('top_k', 0, 1000), ('penalty_last_n', 1, 1048576),
            ('max_images', 1, 4), ('image_max_side', 256, 2048), ('image_jpeg_quality', 30, 100)):
        if type(defaults[name]) is not int or not low <= defaults[name] <= high:
            raise ValueError(f'{name} 必须是 {low} 到 {high} 的整数。')
    for name, low, high in (('temperature', 0, 2), ('top_p', .01, 1), ('min_p', 0, 1), ('repeat_penalty', 1, 2)):
        value = defaults[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} 必须是 {low} 到 {high} 的有限数值。')
    if defaults['max_tokens'] >= defaults['n_ctx']:
        raise ValueError('max_tokens 必须小于 n_ctx，为输入规则和图片保留上下文。')
    return defaults, (_resolve(defaults.get('model_path')), _resolve(defaults.get('mmproj_path')))


def messages(system, user, images, config):
    import numpy as np
    from PIL import Image
    if not isinstance(system, str) or not system.strip():
        raise ValueError('规则不能为空。')
    content = [{'type': 'text', 'text': user}]
    present = [image for image in images if image is not None]
    for image in present:
        if len(image.shape) != 4 or image.shape[0] < 1 or image.shape[-1] not in (3, 4):
            raise ValueError('image 需要 1–4 张 RGB/RGBA 图片，不能超过 max_images。')
    count = sum(image.shape[0] for image in present)
    if count > config['max_images']:
        raise ValueError('所有图片入口合计不能超过 max_images。')
    ordinal = 0
    for image in present:
        for i in range(image.shape[0]):
            ordinal += 1
            array = image[i:i+1][0].detach().cpu().numpy()
            if not np.isfinite(array).all() or min(array.shape[:2]) < 1:
                raise ValueError('图片包含无效像素或尺寸。')
            picture = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8)).convert('RGB')
            picture.thumbnail((config['image_max_side'], config['image_max_side']), Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            picture.save(stream, format='JPEG', quality=config['image_jpeg_quality'])
            data = base64.b64encode(stream.getvalue()).decode('ascii')
            content.extend([{'type': 'text', 'text': f'Image-{ordinal}'},
                            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + data}}])
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}], count


class SXDGemmaImageTask:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'config_override': ('STRING', {'forceInput': True}),
            'system_prompt_override': ('STRING', {'forceInput': True}),
            'user_prompt': ('STRING', {'forceInput': True}),
            'seed': ('INT', {'forceInput': True, 'min': 0, 'max': 0xffffffffffffffff}),
            'task': (['prompt', 'anatomy'], {'default': 'prompt'}),
            'release_after_run': ('BOOLEAN', {'default': False}),
        }, 'optional': {name: ('IMAGE',) for name in ('image', 'image2', 'image3', 'image4')},
            'hidden': {'unique_id': 'UNIQUE_ID'}}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('text', 'raw_output')
    FUNCTION = 'run'
    CATEGORY = 'SXD/Gemma Image'
    DESCRIPTION = '自有 Gemma 图像扩写/大结构检查，与 H3 共用模型缓存；无需 KLL 或外部服务。'

    @classmethod
    def IS_CHANGED(cls, config_override, **kwargs):
        _, paths = parse_config(config_override)
        return repr([runtime._file_identity(path) for path in paths])

    def run(self, config_override, system_prompt_override, user_prompt, seed,
            task='prompt', release_after_run=False, image=None, image2=None, image3=None, image4=None,
            unique_id=None, _request_for_validation=None):
        if task not in ('prompt', 'anatomy', 'spatial_plan'):
            raise ValueError('task 必须为 prompt、anatomy 或内部 spatial_plan。')
        config, paths = parse_config(config_override)
        prepared, count = messages(system_prompt_override, user_prompt,
                                   [image, image2, image3, image4], config)
        if task == 'anatomy' and count != 1:
            raise ValueError('大结构检查必须且只能输入一张成图。')
        if task == 'spatial_plan' and count != 1:
            raise ValueError('结构化规划必须且只能输入一张参考图。')
        request = user_prompt if _request_for_validation is None else _request_for_validation
        trace = Trace(config, validate=lambda text: validate_answer(text, task, request), node_id=unique_id)
        trace.send('starting')
        try:
            result = runtime.H3AuthorGemma4Expand().expand(
                prompt=user_prompt, first_frame=None, frame_count=1, fps=1, seed=seed,
                model=paths[0].name, mmproj=paths[1].name, _model_paths=paths,
                enable_thinking=config['enable_thinking'], max_tokens=config['max_tokens'],
                n_ctx=config['n_ctx'], n_gpu_layers=config['n_gpu_layers'],
                _messages_override=prepared, _author_postprocess=False,
                release_after_expansion=release_after_run, _trace=trace,
                _penalty_last_n=config['penalty_last_n'],
                _sampling={key: config[key] for key in ('temperature', 'top_k', 'top_p', 'min_p', 'repeat_penalty')})
        except BaseException as error:
            trace.send('repetition_detected' if 'THINKING_LOOP' in str(error) else 'error_or_cancelled')
            if isinstance(error, Exception):
                raise RuntimeError('Gemma 图像任务失败：' + str(error) + trace.diagnostic()) from error
            raise
        trace.send('done')
        return result[0], trace.raw_output


class SXDGemmaSpatialPlan(SXDGemmaImageTask):
    """One model call, the same shared runtime and live trace, a typed plan result."""
    @classmethod
    def INPUT_TYPES(cls):
        fields = super().INPUT_TYPES()
        fields['required'] = {
            'config_override': ('STRING', {'forceInput': True}),
            'verified_facts': ('STRING', {'forceInput': True}),
            'user_prompt': ('STRING', {'forceInput': True}),
            'seed': ('INT', {'forceInput': True, 'min': 0, 'max': 0xffffffffffffffff}),
            'release_after_run': ('BOOLEAN', {'default': False}),
        }
        return fields

    RETURN_TYPES = ('STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('plan_json', 'raw_output', 'user_request')
    DESCRIPTION = '单图结构化编辑规划；一次 Gemma 调用。缺姿态、镜头等字段时报错，需接结构化组装节点。'

    @classmethod
    def IS_CHANGED(cls, config_override, **kwargs):
        import hashlib
        return (super().IS_CHANGED(config_override),
                hashlib.sha256(spatial_plan.system_rules().encode('utf-8')).hexdigest())

    def run(self, config_override, verified_facts, user_prompt, seed,
            release_after_run=False, image=None, image2=None, image3=None, image4=None,
            unique_id=None):
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError('原始编辑要求不能为空。')
        if not isinstance(verified_facts, str):
            raise ValueError('已核验图像事实必须是文字，可以留空。')
        content = ('Original editing request:\n' + user_prompt)
        if verified_facts.strip():
            content += '\nVerified reference facts (not new actions):\n' + verified_facts
        result = super().run(config_override, spatial_plan.system_rules(), content, seed,
            task='spatial_plan', release_after_run=release_after_run,
            image=image, image2=image2, image3=image3, image4=image4,
            unique_id=unique_id, _request_for_validation=user_prompt)
        return result[0], result[1], user_prompt


class SXDSpatialPlanAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'plan_json': ('STRING', {'forceInput': True}),
            'user_request': ('STRING', {'forceInput': True}),
        }}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('prompt_json', 'prompt')
    FUNCTION = 'run'
    CATEGORY = 'SXD/Gemma Image'
    DESCRIPTION = '不调用模型；检查人物姿态、物品归属和镜头字段，组装为只有 prompt 的中文 JSON。'

    def run(self, plan_json, user_request):
        return spatial_plan.assemble(plan_json, user_request)

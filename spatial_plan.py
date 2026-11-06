# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate a single-image editing plan and compile it without another model call.

This enforces structure, references and a few literal constraints, not visual truth
or physical feasibility. A missing stance never receives a guessed default.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

POSTURES = {
    'standing': '站立', 'seated': '坐着', 'kneeling': '跪着',
    'crouching': '蹲着', 'lying': '躺卧', 'other': '', 'visible_only': '',
}
ROOT_KEYS = {'version', 'scope', 'goal', 'people', 'objects', 'contacts', 'camera', 'retain'}
PERSON_KEYS = {'id', 'label', 'origin', 'position', 'posture', 'posture_detail',
               'torso_orientation', 'action'}
OBJECT_KEYS = {'id', 'name', 'owner_id', 'mount', 'location'}
CONTACT_KEYS = {'source', 'target', 'relation', 'spatial_detail'}
ENDPOINT_KEYS = {'kind', 'id', 'part'}
CAMERA_KEYS = {'mode', 'direction', 'height', 'framing', 'visibility'}


def _error(path, reason):
    raise ValueError(f'结构化扩写未通过：{path} {reason}。未组装提示词，请检查原始规划输出。')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _error(key, '字段重复')
        result[key] = value
    return result


def _shape(value, keys, path):
    if not isinstance(value, dict):
        _error(path, '必须是对象')
    missing, extra = keys - value.keys(), value.keys() - keys
    if missing or extra:
        _error(path, f'缺少字段 {sorted(missing)}；额外字段 {sorted(extra)}')


def _text(value, path, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        _error(path, '必须是非空文字' if not empty else '必须是文字')
    if len(value) > 2000:
        _error(path, '文字过长')
    if re.search(r'<think>|<\|channel>|<image\d+>|\bImage-\d+\b', value, re.I):
        _error(path, '包含思考标记或单图编号')
    return ' '.join(value.split()).strip()


def _array(value, path, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        _error(path, f'必须是至多 {maximum} 项的数组')


def _enum(value, choices, path):
    if not isinstance(value, str) or value not in choices:
        _error(path, '必须选择 ' + ' / '.join(choices))


def _literal_mount_check(obj, user_request):
    """Guard explicit, literal waist/hand placement; not a semantic interpreter."""
    noun = re.escape(obj['name'])
    waist = re.search(r'(?:腰间|腰部|腰带上)\s*(?:的)?\s*' + noun, user_request)
    handheld = re.search(r'(?:手持|手中(?:拿着|握着)?)\s*(?:的)?\s*' + noun, user_request)
    if waist and not handheld and obj['mount'] != 'waist':
        _error('objects.' + obj['id'] + '.mount', '与原始要求中的腰间归属不符')
    if handheld and not waist and obj['mount'] != 'hand':
        _error('objects.' + obj['id'] + '.mount', '与原始要求中的手持归属不符')


def validate_plan(text, user_request=''):
    if not isinstance(text, str) or len(text) > 100000:
        _error('plan', '需要完整的 JSON 文字且长度不得超过 100000 字符')
    try:
        plan = json.loads(text, object_pairs_hook=_unique,
                          parse_constant=lambda item: _error('plan', '不接受 ' + item))
    except json.JSONDecodeError as error:
        _error('plan', f'JSON 不完整或格式错误（第 {error.lineno} 行）')
    _shape(plan, ROOT_KEYS, 'plan')
    if type(plan['version']) is not int or plan['version'] != 1:
        _error('version', '必须为整数 1')
    _enum(plan['scope'], ('local', 'spatial'), 'scope')
    plan['goal'] = _text(plan['goal'], 'goal')
    plan['retain'] = _text(plan['retain'], 'retain', empty=True)
    for key, maximum in (('people', 12), ('objects', 24), ('contacts', 32)):
        _array(plan[key], key, maximum)
    if plan['scope'] == 'spatial' and not plan['people']:
        _error('people', '空间任务必须填写参与人物')

    people, objects = {}, {}
    for i, person in enumerate(plan['people']):
        path = f'people[{i}]'
        _shape(person, PERSON_KEYS, path)
        for key in PERSON_KEYS:
            person[key] = _text(person[key], path + '.' + key, empty=key == 'posture_detail')
        _enum(person['origin'], ('reference', 'added'), path + '.origin')
        _enum(person['posture'], POSTURES, path + '.posture')
        if person['posture'] == 'visible_only' and person['origin'] != 'reference':
            _error(path + '.posture', '新增人物必须选定目标姿态，不能使用 visible_only')
        if person['posture'] in ('other', 'visible_only', 'kneeling') and not person['posture_detail']:
            _error(path + '.posture_detail', '此姿态需要说明可见状态或具体支撑方式')
        if person['id'] in people:
            _error(path + '.id', '人物编号重复')
        if any(person['label'] == p['label'] for p in people.values()):
            _error(path + '.label', '人物称呼重复，无法区分')
        if any(word in person['label'] for word in ('稳定的人', '新增的人', '参考的人')):
            _error(path + '.label', '请使用自然人物称呼，不输出规则术语')
        people[person['id']] = person

    for i, obj in enumerate(plan['objects']):
        path = f'objects[{i}]'
        _shape(obj, OBJECT_KEYS, path)
        for key in OBJECT_KEYS:
            obj[key] = _text(obj[key], path + '.' + key, empty=key == 'location')
        _enum(obj['mount'], ('waist', 'hand', 'other'), path + '.mount')
        if obj['owner_id'] not in people:
            _error(path + '.owner_id', '没有对应的人物')
        if obj['id'] in objects:
            _error(path + '.id', '物品编号重复')
        if obj['mount'] == 'other' and not obj['location']:
            _error(path + '.location', '必须明确物品位置')
        _literal_mount_check(obj, user_request)
        objects[obj['id']] = obj

    for i, contact in enumerate(plan['contacts']):
        path = f'contacts[{i}]'
        _shape(contact, CONTACT_KEYS, path)
        for key in ('relation', 'spatial_detail'):
            contact[key] = _text(contact[key], path + '.' + key)
        for side in ('source', 'target'):
            endpoint = contact[side]
            _shape(endpoint, ENDPOINT_KEYS, path + '.' + side)
            for key in ENDPOINT_KEYS:
                endpoint[key] = _text(endpoint[key], path + '.' + side + '.' + key)
            _enum(endpoint['kind'], ('person', 'object'), path + '.' + side + '.kind')
            available = people if endpoint['kind'] == 'person' else objects
            if endpoint['id'] not in available:
                _error(path + '.' + side + '.id', '接触端点引用了不存在的人物或物品')
        if contact['source'] == contact['target']:
            _error(path, '接触的两个端点不能完全相同')

    camera = plan['camera']
    _shape(camera, CAMERA_KEYS, 'camera')
    _enum(camera['mode'], ('retain', 'adjust'), 'camera.mode')
    required = plan['scope'] == 'spatial'
    for key in CAMERA_KEYS - {'mode'}:
        camera[key] = _text(camera[key], 'camera.' + key, empty=not required)
    if required:
        if not re.search(r'眼|肩|胸|腰|髋|头|膝|地面|桌面|座面', camera['height']):
            _error('camera.height', '需要明确身体部位或场景高度参照，不能只写与坐姿持平')
        if '侧' in camera['direction'] and not re.search(r'左|右|原图|参考图', camera['direction']):
            _error('camera.direction', '侧向观察需要明确哪一侧，或明确沿用原图的哪一侧')
        if not re.search(r'错开|间隙|空隙|侧|前|后|遮挡|露出|外缘|轮廓|重叠|分开', camera['visibility']):
            _error('camera.visibility', '需要实际位置或遮挡安排，不能只有清晰展示的保证')
    if plan['scope'] == 'local' and (any(p['origin'] == 'added' for p in people.values()) or plan['contacts']):
        _error('scope', '新增人物或接触关系不能标为局部外观修改')
    return plan


def normalized_plan(text, user_request=''):
    return json.dumps(validate_plan(text, user_request), ensure_ascii=False, separators=(',', ':'))


def _clause(value):
    return value.strip().rstrip('。；，,;.!！?？ ')


def assemble(text, user_request=''):
    plan = validate_plan(text, user_request)
    people = {p['id']: p for p in plan['people']}
    objects = {o['id']: o for o in plan['objects']}
    sentences = [_clause(plan['goal'])]
    for person in plan['people']:
        stance = POSTURES[person['posture']]
        parts = [person['label'] + '位于' + _clause(person['position'])]
        if stance:
            parts.append(stance)
        if person['posture_detail']:
            parts.append(_clause(person['posture_detail']))
        parts.extend(['躯干' + _clause(person['torso_orientation']), _clause(person['action'])])
        sentences.append('，'.join(parts))

    def object_name(obj):
        owner = people[obj['owner_id']]['label']
        if obj['mount'] == 'waist':
            return owner + '腰间固定的' + obj['name']
        if obj['mount'] == 'hand':
            return owner + '手持的' + obj['name']
        return owner + '的' + obj['name']

    for obj in plan['objects']:
        if obj['location']:
            sentences.append(object_name(obj) + '位于' + _clause(obj['location']))

    def endpoint(value):
        subject = people[value['id']]['label'] if value['kind'] == 'person' else object_name(objects[value['id']])
        return subject + '的' + _clause(value['part'])

    for contact in plan['contacts']:
        sentences.append(endpoint(contact['source']) + _clause(contact['relation']) +
                         endpoint(contact['target']) + '，' + _clause(contact['spatial_detail']))
    camera = plan['camera']
    if any(camera[key] for key in ('direction', 'height', 'framing', 'visibility')):
        clauses = [('沿用参考图视角，' if camera['mode'] == 'retain' else '') +
                   '镜头' + _clause(camera['direction'])]
        clauses.extend(_clause(camera[key]) for key in ('height', 'framing', 'visibility') if camera[key])
        sentences.append('，'.join(clauses))
    if plan['retain']:
        sentences.append(_clause(plan['retain']))
    prompt = '。'.join(item for item in sentences if item) + '。'
    if not re.search(r'[\u3400-\u9fff]', prompt):
        _error('prompt', '必须使用中文描述')
    return json.dumps({'prompt': prompt}, ensure_ascii=False, separators=(',', ':')), prompt


def system_rules():
    return Path(__file__).with_name('spatial_plan_rules.txt').read_text(encoding='utf-8')

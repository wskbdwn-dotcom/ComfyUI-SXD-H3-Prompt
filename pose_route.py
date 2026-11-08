# SPDX-License-Identifier: GPL-3.0-or-later
"""Choose one packaged pose entry, or stop. The expansion model does not choose."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

CONTACTS = (
    '阴茎进入阴道',
    '只进入肛门',
    '嘴接触阴茎',
    '嘴接触阴部',
    '互相口爱',
    '足接触阴茎',
    '手指或玩具接触自己的阴蒂',
)
POSE_RULES = """你是图像编辑指令写作者。只输出简体中文，一行 JSON，且只有一个非空字符串字段 prompt。不要 Markdown，不要解释。

用户消息里有一条已经取定的姿势。提示词必须原样保留这条里的承重、胸的朝向、站位、接触和镜头。不要另选姿势，不要改这些身体关系。
从参考图和原始编辑要求里补上身份、衣服和场景。不要编造姿势库里没有写出的支撑或机位。
"""
STORE = Path(__file__).resolve().parent / 'pose_data' / 'entries'
MAX_ZIP_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.avif'}


class PoseStop(Exception):
    """The pose branch has no legal entry. Callers must not fall back to free expansion."""


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'JSON 字段重复：{key}')
        result[key] = value
    return result


def parse_contact(text):
    """Return a closed contact record. Extra fields and unknown labels fail."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('接触判断结果为空。')
    value = json.loads(text, object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError('接触判断必须是 JSON 对象。')
    route = value.get('route')
    if route == 'general':
        if set(value) != {'route'}:
            raise ValueError('不交合时只能有 route 字段。')
        return {'route': 'general', 'contacts': [], 'named_pose': '',
                'people': None, 'vertical': None, 'face_visible': None}
    if route != 'pose':
        raise ValueError('route 只能是 general 或 pose。')
    expected = {'route', 'contacts', 'named_pose', 'people', 'vertical', 'face_visible'}
    if set(value) != expected:
        raise ValueError('交合判断的字段必须是 route、contacts、named_pose、people、vertical、face_visible。')
    contacts = value['contacts']
    if not isinstance(contacts, list) or len(contacts) > 3 or len(set(contacts)) != len(contacts):
        raise ValueError('contacts 必须是至多 3 个不重复的接触类别。')
    if any(item not in CONTACTS for item in contacts):
        raise ValueError('contacts 含有姿势库不使用的类别。')
    ordered = [item for item in CONTACTS if item in contacts]
    named = value['named_pose']
    if not isinstance(named, str) or len(named) > 80:
        raise ValueError('named_pose 必须是不超过 80 字的字符串，没有点名就留空。')
    people = value['people']
    if type(people) is not int or not 1 <= people <= 4:
        raise ValueError('people 必须是 1 到 4 的整数。')
    for name in ('vertical', 'face_visible'):
        if type(value[name]) is not bool:
            raise ValueError(name + ' 必须是布尔值。')
    return {'route': 'pose', 'contacts': ordered, 'named_pose': named.strip(),
            'people': people, 'vertical': value['vertical'], 'face_visible': value['face_visible']}


def contacts_in_sentence(sentence):
    if not isinstance(sentence, str):
        return []
    found = []
    for label in CONTACTS:
        if label == '只进入肛门' and ('进入肛门' in sentence or '只进入肛门' in sentence):
            found.append(label)
        elif label == '阴茎进入阴道' and (
                '阴茎进入阴道' in sentence or ('阴道' in sentence and ('进入' in sentence or '浅入' in sentence) and '肛门' not in sentence)):
            found.append(label)
        elif label == '手指或玩具接触自己的阴蒂' and '阴蒂' in sentence and ('手指' in sentence or '玩具' in sentence):
            found.append(label)
        elif label in sentence:
            found.append(label)
    return found


def load_library(directory):
    path = Path(directory)
    if not path.is_dir():
        if path == STORE:
            raise PoseStop('还没有上传姿势数据。')
        raise PoseStop('姿势库目录不存在：' + str(path))
    entries = []
    for file in sorted(path.glob('pose_*.json')):
        if not file.is_file():
            continue
        data = json.loads(file.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not data.get('id'):
            raise PoseStop('姿势条目缺少 id：' + file.name)
        priority = data.get('selection_priority')
        if priority is not None and (type(priority) is not int or priority < 0):
            raise PoseStop(data['id'] + ' 的 selection_priority 必须是非负整数。')
        sentence = data.get('pose_sentence') if isinstance(data.get('pose_sentence'), str) else ''
        entries.append({
            'id': data['id'],
            'name': data.get('name') if isinstance(data.get('name'), str) else '',
            'alias': data.get('alias') if isinstance(data.get('alias'), str) else '',
            'usable': data.get('usable_for_selection') is True and bool(sentence.strip()),
            'priority': priority,
            'contacts': contacts_in_sentence(sentence),
            'pose_sentence': sentence.strip(),
        })
    if not entries:
        raise PoseStop('姿势库目录里没有条目：' + str(path))
    return entries


def library_token(directory):
    path = Path(directory)
    if not path.is_dir():
        return 'missing'
    files = sorted(item for item in path.glob('pose_*.json') if item.is_file())
    return str([(item.name, item.stat().st_mtime_ns, item.stat().st_size) for item in files])


def _rows(entries):
    return [{
        'id': entry['id'], 'name': entry['name'], 'alias': entry['alias'],
        'usable': entry['usable'], 'priority': entry['priority'],
        'contacts': entry['contacts'], 'pose_sentence': entry['pose_sentence'],
    } for entry in entries]


def lookup(entries, query):
    """Return the chosen entry, a status, and every entry the query matched."""
    named = query['named_pose']
    if named:
        hits = [entry for entry in entries if named in (entry['name'], entry['alias'])]
        if not hits:
            return None, f'查无此条：姿势名「{named}」不在库里。已停止，不自由扩写。', []
        if len(hits) > 1:
            return None, '同名姿势超过一条，已停止，不自由扩写。', _rows(hits)
        entry = hits[0]
        if not entry['usable']:
            return None, f'查无此条：{entry["id"]} 还不能选取。已停止，不自由扩写。', _rows(hits)
        if query['contacts'] and set(entry['contacts']) != set(query['contacts']):
            return None, f'查无此条：{entry["id"]} 完成不了要求的接触。已停止，不自由扩写。', _rows(hits)
        return entry, f'交合：{entry["id"]} {entry["alias"] or entry["name"]}', _rows(hits)
    if not query['contacts']:
        return None, '查无此条：没有点名，也没有可匹配的接触。已停止，不自由扩写。', []
    hits = [entry for entry in entries if set(entry['contacts']) == set(query['contacts'])]
    usable = [entry for entry in hits if entry['usable']]
    if not usable:
        message = '查无此条：没有可选取的姿势能完成这个接触。已停止，不自由扩写。'
        return None, message, _rows(hits)
    if len(usable) == 1:
        entry = usable[0]
        return entry, f'交合：{entry["id"]} {entry["alias"] or entry["name"]}', _rows(hits)
    if any(entry['priority'] is None for entry in usable) or len({entry['priority'] for entry in usable}) != len(usable):
        return None, '同一接触有多条可选取姿势，默认顺序还没排好。已停止，不自由扩写。', _rows(hits)
    entry = min(usable, key=lambda item: item['priority'])
    return entry, f'交合：{entry["id"]} {entry["alias"] or entry["name"]}', _rows(hits)


def prepare(contact_json, original_request, general_rules, pose_rules, library_dir):
    if not isinstance(original_request, str) or not original_request.strip():
        raise ValueError('原始编辑要求不能为空。')
    if not isinstance(general_rules, str) or not general_rules.strip():
        raise ValueError('通用扩写规则不能为空。')
    query = parse_contact(contact_json)
    extraction = json.dumps(query, ensure_ascii=False, indent=2)
    if query['route'] == 'general':
        return {'ok': True, 'system': general_rules, 'user': original_request,
                'status': '不交合：沿用通用扩写规则', 'extraction': extraction, 'matches': '[]'}
    if not isinstance(pose_rules, str) or not pose_rules.strip():
        raise ValueError('交合扩写规则不能为空。')
    try:
        entries = load_library(library_dir)
    except PoseStop as error:
        return {'ok': False, 'system': '', 'user': '', 'status': str(error),
                'extraction': extraction, 'matches': '[]'}
    entry, status, rows = lookup(entries, query)
    shown = json.dumps(rows, ensure_ascii=False, indent=2)
    if entry is None:
        return {'ok': False, 'system': '', 'user': '', 'status': status,
                'extraction': extraction, 'matches': shown}
    user = (original_request.rstrip()
            + '\n\n已取定姿势。下面的摆法必须原样写入提示词，不得改支撑、站位、接触和镜头：\n'
            + entry['pose_sentence'])
    return {'ok': True, 'system': pose_rules, 'user': user, 'status': status,
            'extraction': extraction, 'matches': shown}


def install_zip(payload, store=STORE):
    """Replace the saved pose JSON with the text entries inside one zip."""
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ValueError('没有读到压缩包。')
    if len(payload) > MAX_ZIP_BYTES:
        raise ValueError('压缩包超过 32MB。')
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError('这不是 zip 压缩包。') from error
    collected = []
    with archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        if not files:
            raise ValueError('压缩包里没有文件。')
        if len(files) > 5000:
            raise ValueError('压缩包里的文件超过 5000 个。')
        for info in files:
            filename = Path(info.filename).name
            suffix = Path(filename).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                raise ValueError('压缩包里有图片。只上传文本 JSON。')
            if suffix != '.json' or not filename or filename.startswith('.'):
                raise ValueError('只接受 .json 文件：' + info.filename)
            if info.file_size > MAX_JSON_BYTES:
                raise ValueError(filename + ' 超过 2MB。')
            try:
                data = json.loads(archive.read(info).decode('utf-8'))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(filename + ' 不是 UTF-8 JSON。') from error
            items = data if isinstance(data, list) else [data]
            if not isinstance(data, (dict, list)):
                raise ValueError(filename + ' 必须是 JSON 对象或对象列表。')
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'].strip():
                    raise ValueError(filename + ' 里有条目缺少 id。')
                pose_id = item['id'].strip()
                if not pose_id.replace('_', '').isalnum():
                    raise ValueError('id 只能包含字母、数字和下划线：' + pose_id)
                collected.append((pose_id, item))
    if not collected:
        raise ValueError('压缩包里没有姿势条目。')
    seen = set()
    for pose_id, _item in collected:
        if pose_id in seen:
            raise ValueError('重复 id：' + pose_id)
        seen.add(pose_id)
    store.mkdir(parents=True, exist_ok=True)
    for old in store.glob('pose_*.json'):
        old.unlink()
    for pose_id, item in collected:
        filename = pose_id if pose_id.startswith('pose_') else 'pose_' + pose_id
        (store / f'{filename}.json').write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return len(collected)

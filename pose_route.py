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
WEIGHT_CONTACT = 60
WEIGHT_WOMAN_CHEST = 50
WEIGHT_MAN_CHEST = 30
WEIGHT_POSTURE = 80
WEIGHT_AXIS = 50
WEIGHT_HEAD = 15
CHEST_PHRASES = (
    ('胸朝向她的背', '朝向她的背'),
    ('胸朝向他的背', '朝向他的背'),
    ('胸朝向女人的背', '朝向她的背'),
    ('胸朝向男人的背', '朝向他的背'),
    ('胸朝向对方', '朝向对方'),
    ('胸朝向女人', '朝向对方'),
    ('胸朝向男人', '朝向对方'),
    ('胸背对对方', '背对对方'),
    ('胸背对女人', '背对对方'),
    ('胸背对男人', '背对对方'),
    ('背对对方', '背对对方'),
    ('背对女人', '背对对方'),
    ('背对男人', '背对对方'),
    ('胸朝上', '朝上'),
    ('胸朝下', '朝下'),
)
POSTURE_PHRASES = (
    ('坐在对方身上', '坐在对方身上'),
    ('坐在他身上', '坐在对方身上'),
    ('坐在她身上', '坐在对方身上'),
    ('坐在男人身上', '坐在对方身上'),
    ('坐在女人身上', '坐在对方身上'),
    ('仰躺', '仰躺'),
    ('侧卧', '侧卧'),
    ('仰卧', '仰躺'),
    ('侧躺', '侧卧'),
    ('坐在椅子上', '坐'),
    ('坐在沙发上', '坐'),
    ('趴着', '扒'),
    ('趴下', '扒'),
    ('坐着', '坐'),
    ('坐下', '坐'),
    ('坐姿', '坐'),
    ('跪着', '跪'),
    ('跪下', '跪'),
    ('跪姿', '跪'),
    ('站着', '站'),
    ('站立', '站'),
    ('站姿', '站'),
    ('扒', '扒'),
    ('趴', '扒'),
    ('跪', '跪'),
    ('站', '站'),
    ('坐', '坐'),
)
AXIS_PHRASES = (
    '男方在女方身后', '男人在女人身后', '男方在她身后', '男人在她身后',
    '在她身后', '在女人身后', '在她后面', '在女人后面', '她后面', '女人后面',
    '在她背后', '在女人背后', '从后面', '从后方',
)
HEAD_PHRASES = (
    '头朝向他的骨盆', '头朝向她的骨盆', '头朝向他的头', '头朝向她的脸',
    '头朝向他的胯', '头朝向她的胯', '头转向他', '头转向她', '头朝向他', '头朝向她',
    '头后仰', '头低下', '头朝上', '头朝下', '头朝前',
)


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
            'people': people, 'vertical': value['vertical'], 'face_visible': value['face_visible'],
            'schema': 'contact'}


SLOT_NAMES = ('头', '胸', '腰', '胯', '臀', '大腿', '膝盖', '脚')
POSTURE_ORDER = ('站', '跪', '扒', '仰躺', '侧卧', '坐', '坐在对方身上')
WAIST_PHRASES = ('腰略后仰', '腰前折', '腰后仰', '腰侧扭', '腰直')
HIP_PHRASES = (
    '胯朝向他的脸', '胯朝向她的脸', '胯朝向女人的脸', '胯朝向男人的脸',
    '胯朝向对方', '胯朝向他', '胯朝向她', '胯朝向女人', '胯朝向男人',
    '胯转向一侧', '胯朝上', '胯朝前',
)
BUTT_PHRASES = (
    '臀朝向他的脸', '臀朝向她的脸', '臀坐在他身上', '臀坐在她身上',
    '臀坐在男人身上', '臀坐在女人身上', '臀承在水平承重面上',
    '臀朝向他', '臀朝向她', '臀朝向女人', '臀朝向男人', '臀抬起',
)
THIGH_PHRASES = ('大腿分开', '大腿并拢', '大腿伸直', '大腿抬起')
KNEE_PHRASES = ('膝盖跪在水平承重面上', '膝盖弯曲', '膝盖伸直', '膝盖着地')
FOOT_PHRASES = ('脚着地', '脚抬起', '脚并拢')
CLOSED_SLOTS = {
    '头': (),
    '胸': ('朝上', '朝下', '朝向对方', '背对对方', '朝向她的背', '朝向他的背'),
    '腰': WAIST_PHRASES,
    '胯': HIP_PHRASES,
    '臀': BUTT_PHRASES,
    '大腿': THIGH_PHRASES,
    '膝盖': KNEE_PHRASES,
    '脚': FOOT_PHRASES,
}


def _clean(text):
    if not isinstance(text, str):
        return ''
    return text.strip().strip('。．. \t\r\n')


def _closed(text, phrases):
    text = _clean(text)
    return text if text in phrases else ''


def _slots(raw):
    if not isinstance(raw, dict):
        raise ValueError('body_parts 的每一方必须是对象。')
    slots = {}
    for name in SLOT_NAMES:
        value = raw.get(name, '')
        if value is None:
            value = ''
        if not isinstance(value, str):
            raise ValueError('身体槽必须是字符串：' + name)
        if name == '胸':
            slots[name] = _closed(value, CLOSED_SLOTS['胸']) or _first(value, CHEST_PHRASES)
        elif name == '头':
            slots[name] = _first(value, HEAD_CLASSES) if value else ''
        else:
            slots[name] = _closed(value, CLOSED_SLOTS[name])
    return slots


def parse_extraction(text):
    """Normalize the demo extraction document. An empty contact is the general route."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('原话提取结果为空。')
    value = json.loads(text, object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError('原话提取必须是 JSON 对象。')
    required = {'人数', 'alias', '接触', '支撑', '轴线', 'body_parts'}
    missing = required - set(value)
    if missing:
        raise ValueError('原话提取缺少字段：' + '、'.join(sorted(missing)))
    people = value['人数']
    if type(people) is not int or not 0 <= people <= 20:
        people = 0
    alias = value['alias'] if isinstance(value['alias'], str) else ''
    alias = alias.strip()
    if len(alias) > 80:
        raise ValueError('alias 超过 80 字。')
    contact = _closed(value['接触'] if isinstance(value['接触'], str) else '', CONTACTS)
    posture = _first(value['支撑'] if isinstance(value['支撑'], str) else '', POSTURE_PHRASES)
    axis_text = value['轴线'] if isinstance(value['轴线'], str) else ''
    axis = '男方在女方身后' if _clean(axis_text) == '男方在女方身后' or _axis(axis_text) else ''
    parts = value['body_parts']
    if not isinstance(parts, dict) or not {'女方', '男方'} <= set(parts):
        raise ValueError('body_parts 必须包含女方和男方。')
    woman = _slots(parts['女方'])
    man = _slots(parts['男方'])
    if woman['胸']:
        woman_chest = woman['胸']
    else:
        woman_chest = ''
    document = {
        '人数': people, 'alias': alias, '接触': contact, '支撑': posture, '轴线': axis,
        'body_parts': {'女方': woman, '男方': man},
    }
    route = 'pose' if contact else 'general'
    return {
        'route': route, 'schema': 'extract', 'contacts': [contact] if contact else [],
        'named_pose': alias, 'people': people, 'vertical': None, 'face_visible': None,
        'woman_chest': woman_chest, 'man_chest': man['胸'], 'posture': posture,
        'axis': axis, 'head': woman['头'], 'document': document,
    }


def parse_image_state(text):
    """Normalize the image-support document. Empty posture means the pose is not ready."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError('读图结果为空。')
    value = json.loads(text, object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError('读图结果必须是 JSON 对象。')
    if set(value) != {'支撑', '姿势已准备', '事实'}:
        raise ValueError('读图结果的字段必须是支撑、姿势已准备、事实。')
    posture = _first(value['支撑'] if isinstance(value['支撑'], str) else '', POSTURE_PHRASES)
    ready = value['姿势已准备'] is True
    facts = value['事实']
    if not isinstance(facts, list):
        raise ValueError('事实必须是字符串数组。')
    lines = []
    for item in facts:
        if not isinstance(item, str):
            raise ValueError('事实里的每一条都必须是字符串。')
        line = item.strip()
        if not line or line in lines:
            continue
        lines.append(line[:120])
        if len(lines) == 8:
            break
    if not posture:
        ready = False
    return {'支撑': posture, '姿势已准备': ready, '事实': lines}


def choose_support(user_posture, image=None):
    """User posture wins. Otherwise use the picture. Otherwise keep every posture."""
    user = _first(user_posture, POSTURE_PHRASES) if user_posture else ''
    if user:
        return {'支撑': user, '来源': '用户', '直接适配': False, '备选全部支撑': False,
                '说明': '原话写了支撑，按原话选。'}
    if not image:
        return {'支撑': '', '来源': '', '直接适配': False, '备选全部支撑': True,
                '说明': '原话没写支撑。每种支撑都作备选。'}
    img = image.get('支撑') or ''
    ready = image.get('姿势已准备') is True
    if img and ready:
        return {'支撑': img, '来源': '图片', '直接适配': True, '备选全部支撑': False,
                '说明': '图里已经摆好这个姿势。原话没写支撑，按图里的姿势直接适配。'}
    if img == '站':
        message = '原话没写支撑。图里的人站着，只选站立的姿势。'
    elif img == '坐':
        message = '原话没写支撑。图里的人坐着，只选坐姿。'
    elif img:
        message = '原话没写支撑。按图里的支撑选。'
    elif ready:
        return {'支撑': '', '来源': '图片', '直接适配': False, '备选全部支撑': True,
                '说明': '图里像是已经在摆姿势，但看不出支撑。每种支撑都作备选。'}
    else:
        return {'支撑': '', '来源': '', '直接适配': False, '备选全部支撑': True,
                '说明': '看不出支撑。每种支撑都作备选。'}
    return {'支撑': img, '来源': '图片', '直接适配': False, '备选全部支撑': False, '说明': message}


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


def _by_length(table):
    return tuple(sorted(table, key=lambda item: len(item[0]) if isinstance(item, tuple) else len(item), reverse=True))


CHEST_PHRASES = _by_length(CHEST_PHRASES)
POSTURE_PHRASES = _by_length(POSTURE_PHRASES)
AXIS_PHRASES = _by_length(tuple((item, item) for item in AXIS_PHRASES))
HEAD_CLASSES = _by_length(tuple(
    (phrase, phrase) for phrase in HEAD_PHRASES
) + tuple(
    (phrase.replace('她', '女人').replace('他', '男人'), phrase)
    for phrase in HEAD_PHRASES
    if phrase.replace('她', '女人').replace('他', '男人') != phrase
))


def _first(text, table):
    if not text:
        return ''
    for phrase, label in table:
        if phrase and phrase in text:
            return label
    return ''


def _person_spans(sentence):
    current = '女'
    woman, man = [], []
    for part in sentence.split('。'):
        text = part.strip()
        if not text:
            continue
        if text.startswith(('男方', '男人')):
            current = '男'
        elif text.startswith(('女方', '女人')):
            current = '女'
        (man if current == '男' else woman).append(text)
    return '。'.join(woman), '。'.join(man)


def _axis(text):
    return '男方在女方身后' if _first(text, AXIS_PHRASES) else ''


def _posture_fits(query, pose):
    if not query:
        return False
    if query == pose:
        return True
    return query == '坐' and pose == '坐在对方身上'


def _entry_labels(sentence):
    woman, man = _person_spans(sentence)
    return {
        'woman_chest': _first(woman, CHEST_PHRASES),
        'man_chest': _first(man, CHEST_PHRASES),
        'posture': _first(woman, POSTURE_PHRASES),
        'axis': _axis(sentence),
        'head': _first(woman, HEAD_CLASSES),
    }


def _request_labels(text):
    """Read only the closed labels the request already states. Empty labels do not score."""
    axis = _axis(text)
    woman_chest = ''
    man_chest = ''
    if '她的背' in text or '女人的背' in text:
        man_chest = '朝向她的背'
    if '他的背' in text or '男人的背' in text:
        woman_chest = '朝向他的背'
    elif any(phrase in text for phrase in ('背对男人', '背对他', '胸背对男人', '背对对方', '胸背对对方')):
        woman_chest = '背对对方'
    elif any(phrase in text for phrase in ('胸朝向男人', '胸朝向他', '胸朝向对方')):
        woman_chest = '朝向对方'
    elif '胸朝下' in text:
        woman_chest = '朝下'
    elif '胸朝上' in text:
        woman_chest = '朝上'
    if axis and not woman_chest:
        woman_chest = '背对对方'
    return {
        'woman_chest': woman_chest,
        'man_chest': man_chest,
        'posture': _first(text, POSTURE_PHRASES),
        'axis': axis,
        'head': _first(text, HEAD_CLASSES),
    }


def _stated_match(labels, entry):
    if labels['woman_chest'] and entry['woman_chest'] != labels['woman_chest']:
        return False
    if labels['man_chest'] and entry['man_chest'] != labels['man_chest']:
        return False
    if labels['axis'] and entry['axis'] != labels['axis']:
        return False
    if labels['posture'] and not _posture_fits(labels['posture'], entry['posture']):
        return False
    return True


def _score(query, labels, entry):
    points = WEIGHT_CONTACT if query['contacts'] else 0
    if labels['woman_chest']:
        points += WEIGHT_WOMAN_CHEST if entry['woman_chest'] == labels['woman_chest'] else 0
    if labels['man_chest']:
        points += WEIGHT_MAN_CHEST if entry['man_chest'] == labels['man_chest'] else 0
    if labels['posture']:
        points += WEIGHT_POSTURE if _posture_fits(labels['posture'], entry['posture']) else 0
    if labels['axis']:
        points += WEIGHT_AXIS if entry['axis'] == labels['axis'] else 0
    if labels['head']:
        points += WEIGHT_HEAD if entry['head'] == labels['head'] else 0
    return points


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
        sentence = data.get('pose_sentence') if isinstance(data.get('pose_sentence'), str) else ''
        number = data.get('number')
        if type(number) is not int:
            number = 10**9
        entries.append({
            'id': data['id'],
            'number': number,
            'name': data.get('name') if isinstance(data.get('name'), str) else '',
            'alias': data.get('alias') if isinstance(data.get('alias'), str) else '',
            'usable': data.get('usable_for_selection') is True and bool(sentence.strip()),
            'contacts': contacts_in_sentence(sentence),
            'pose_sentence': sentence.strip(),
            **_entry_labels(sentence),
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
        'id': entry['id'], 'number': entry['number'], 'name': entry['name'], 'alias': entry['alias'],
        'usable': entry['usable'], 'score': entry.get('score', 0),
        'contacts': entry['contacts'], 'woman_chest': entry['woman_chest'], 'man_chest': entry['man_chest'],
        'posture': entry['posture'], 'axis': entry['axis'], 'head': entry['head'],
        'pose_sentence': entry['pose_sentence'],
    } for entry in entries]


def _rank_key(item):
    points, entry = item
    return (-points, entry['number'], entry['id'])


def _spread_rows(passed):
    """One highest-scoring finalized entry for each posture, in the fixed order."""
    chosen = []
    seen = set()
    for posture in POSTURE_ORDER:
        if posture == '坐':
            group = [item for item in passed if item[1]['posture'] in ('坐', '坐在对方身上')]
        else:
            group = [item for item in passed if item[1]['posture'] == posture]
        if not group:
            continue
        group.sort(key=_rank_key)
        if group[0][1]['id'] not in seen:
            chosen.append(group[0])
            seen.add(group[0][1]['id'])
    unlabeled = [item for item in passed if item[1]['id'] not in seen]
    unlabeled.sort(key=_rank_key)
    if unlabeled:
        chosen.append(unlabeled[0])
    return chosen


def lookup(entries, query, request='', labels=None, choice=None):
    """Score finalized entries. Image conflicts are not read at runtime."""
    named = query['named_pose']
    if labels is None:
        labels = _request_labels(request)
    else:
        labels = dict(labels)
    if choice and choice.get('备选全部支撑'):
        labels['posture'] = ''
    elif choice and choice.get('支撑'):
        labels['posture'] = choice['支撑']
    if named:
        hits = [entry for entry in entries if named in (entry['name'], entry['alias'])]
        if not hits:
            return None, f'查无此条：姿势名「{named}」不在库里。已停止，不自由扩写。', []
        if len(hits) > 1:
            return None, '同名姿势超过一条，已停止，不自由扩写。', _rows(hits)
        entry = hits[0]
        if not entry['usable']:
            return None, f'查无此条：{entry["id"]} 还不是定稿。已停止，不自由扩写。', []
        if query['contacts'] and set(entry['contacts']) != set(query['contacts']):
            return None, f'查无此条：{entry["id"]} 完成不了要求的接触。已停止，不自由扩写。', _rows(hits)
        entry['score'] = _score(query, labels, entry)
        return entry, f'交合：{entry["id"]} {entry["alias"] or entry["name"]}，得分 {entry["score"]}', _rows(hits)
    if not query['contacts']:
        return None, '查无此条：没有点名，也没有可匹配的接触。已停止，不自由扩写。', []
    usable = [entry for entry in entries if entry['usable'] and set(entry['contacts']) == set(query['contacts'])]
    if not usable:
        return None, '查无此条：没有定稿姿势能完成这个接触。已停止，不自由扩写。', []
    passed = []
    for entry in usable:
        entry['score'] = _score(query, labels, entry)
        if _stated_match(labels, entry):
            passed.append((entry['score'], entry))
    if not passed:
        return None, '查无此条：定稿姿势对不上已经写明的朝向、支撑或轴线。已停止，不自由扩写。', _rows(usable)
    spread = bool(choice and choice.get('备选全部支撑'))
    if spread:
        passed = _spread_rows(passed)
    else:
        passed.sort(key=_rank_key)
        passed = passed[:3]
    if not passed:
        return None, '查无此条：定稿姿势对不上已经写明的朝向、支撑或轴线。已停止，不自由扩写。', _rows(usable)
    entry = passed[0][1]
    ordered = [item[1] for item in passed]
    return entry, f'交合：{entry["id"]} {entry["alias"] or entry["name"]}，得分 {entry["score"]}', _rows(ordered)


def _blank_support():
    return {'支撑': '', '来源': '', '直接适配': False, '备选全部支撑': True, '说明': '还没有读图结果。'}


def prepare(contact_json, original_request, general_rules, pose_rules, library_dir, image_json=''):
    if not isinstance(original_request, str) or not original_request.strip():
        raise ValueError('原始编辑要求不能为空。')
    if not isinstance(general_rules, str) or not general_rules.strip():
        raise ValueError('通用扩写规则不能为空。')
    query = parse_pose_input(contact_json)
    image = parse_image_state(image_json) if isinstance(image_json, str) and image_json.strip() else None
    if query.get('schema') == 'extract':
        labels = {key: query[key] for key in ('woman_chest', 'man_chest', 'posture', 'axis', 'head')}
        choice = choose_support(query['posture'], image)
        extraction = json.dumps(query['document'], ensure_ascii=False, indent=2)
    else:
        labels = _request_labels(original_request)
        choice = choose_support(labels['posture'], image)
        extraction = json.dumps(query, ensure_ascii=False, indent=2)
    image_state = json.dumps(image, ensure_ascii=False, indent=2) if image else ''
    support = json.dumps(choice, ensure_ascii=False, indent=2)
    if query['route'] == 'general':
        return {'ok': True, 'system': general_rules, 'user': original_request,
                'status': '不交合：沿用通用扩写规则', 'extraction': extraction, 'matches': '[]',
                'support': support, 'image_state': image_state}
    if not isinstance(pose_rules, str) or not pose_rules.strip():
        raise ValueError('交合扩写规则不能为空。')
    try:
        entries = load_library(library_dir)
    except PoseStop as error:
        return {'ok': False, 'system': '', 'user': '', 'status': str(error),
                'extraction': extraction, 'matches': '[]', 'support': support, 'image_state': image_state}
    entry, status, rows = lookup(entries, query, original_request, labels, choice)
    shown = json.dumps(rows, ensure_ascii=False, indent=2)
    if entry is None:
        return {'ok': False, 'system': '', 'user': '', 'status': status,
                'extraction': extraction, 'matches': shown, 'support': support, 'image_state': image_state}
    user = (original_request.rstrip()
            + '\n\n已取定姿势。下面的摆法必须原样写入提示词，不得改支撑、站位、接触和镜头：\n'
            + entry['pose_sentence'])
    return {'ok': True, 'system': pose_rules, 'user': user, 'status': status,
            'extraction': extraction, 'matches': shown, 'support': support, 'image_state': image_state}


def parse_pose_input(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('原话提取结果为空。')
    value = json.loads(text, object_pairs_hook=_unique)
    if isinstance(value, dict) and '接触' in value:
        return parse_extraction(text)
    return parse_contact(text)


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

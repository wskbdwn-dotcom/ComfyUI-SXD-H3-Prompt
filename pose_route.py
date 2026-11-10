# SPDX-License-Identifier: GPL-3.0-or-later
"""Choose one packaged pose entry, or stop. The expansion model does not choose."""
from __future__ import annotations

import io
import json
import random
import re
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
POSE_RULES = """你是图像编辑指令写作者。只输出简体中文，一个 JSON 对象，且只有一个非空字符串字段 prompt。不要 Markdown，不要解释。

用户消息里有一条已经取定的姿势。不要另选姿势，不要改这条里的承重、胸的朝向、站位和接触。
有姿势图时，prompt 的正文必须是下面这五段，顺序不变，段与段之间换行。前三行逐字照抄：

图1是女人的形象。
图2是姿势参考图。
目标是保持图1的背景，让图1女人完成图2的姿势。
生图详细描述：
镜头：

生图详细描述写成连贯的话，不要分点。先写图1和原话里已经有的衣着、身材、场景，再写入已取定的摆法。图2只提供身体摆法，不要把图2里的脸、发型、衣服或房间写进来，也不要换掉图1的背景。
镜头这一行：原话写了景别或取景范围就用原话；没写就用已取定的镜头。没有镜头素材时不要自己编景别。
"""
STORE = Path(__file__).resolve().parent / 'pose_data' / 'entries'
IMAGE_DIR = 'images'
MAX_ZIP_BYTES = 128 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.avif'}
WEIGHT_CONTACT = 60
WEIGHT_WOMAN_CHEST = 50
WEIGHT_MAN_CHEST = 30
WEIGHT_POSTURE = 80
WEIGHT_AXIS = 50
WEIGHT_HEAD = 15
DETAIL_WEIGHTS = (('腰', 8), ('大腿', 6), ('膝盖', 5), ('胯', 4), ('臀', 3), ('脚', 2))
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
    ('被抱离地面', '被抱起'),
    ('被抱起来', '被抱起'),
    ('抱离地面', '被抱起'),
    ('被抱起', '被抱起'),
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
POSTURE_ORDER = ('站', '跪', '扒', '仰躺', '侧卧', '坐', '坐在对方身上', '被抱起')
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


def _slot_text(value):
    """A slot is one phrase. A one-item list is the same phrase, not a failure."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return texts[0] if texts else ''
    raise ValueError('身体槽必须是字符串。')


def _slots(raw):
    if not isinstance(raw, dict):
        raise ValueError('body_parts 的每一方必须是对象。')
    slots = {}
    for name in SLOT_NAMES:
        try:
            value = _slot_text(raw.get(name, ''))
        except ValueError:
            raise ValueError('身体槽必须是字符串：' + name) from None
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
    split = '女方支撑' in value or '男方支撑' in value
    required = {'人数', 'alias', '接触', '轴线', 'body_parts'}
    if not split:
        required.add('支撑')
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
    if split:
        posture = _first(value.get('女方支撑') if isinstance(value.get('女方支撑'), str) else '', POSTURE_PHRASES)
        man_posture = _first(value.get('男方支撑') if isinstance(value.get('男方支撑'), str) else '', POSTURE_PHRASES)
    else:
        posture = _first(value.get('支撑') if isinstance(value.get('支撑'), str) else '', POSTURE_PHRASES)
        man_posture = ''
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
        '人数': people, 'alias': alias, '接触': contact, '支撑': posture,
        '女方支撑': posture, '男方支撑': man_posture, '轴线': axis,
        'body_parts': {'女方': woman, '男方': man},
    }
    route = 'pose' if contact else 'general'
    return {
        'route': route, 'schema': 'extract', 'contacts': [contact] if contact else [],
        'named_pose': alias, 'people': people, 'vertical': None, 'face_visible': None,
        'woman_chest': woman_chest, 'man_chest': man['胸'], 'posture': posture,
        'woman_posture': posture, 'man_posture': man_posture,
        'axis': axis, 'head': woman['头'],
        'woman_parts': _detail_parts(woman), 'man_parts': _detail_parts(man),
        'document': document,
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
        message = '原话没写女方支撑。图里的人站着，这种支撑优先。'
    elif img == '坐':
        message = '原话没写女方支撑。图里的人坐着，这种支撑优先。'
    elif img:
        message = '原话没写女方支撑。按图里的支撑优先。'
    elif ready:
        return {'支撑': '', '来源': '图片', '直接适配': False, '备选全部支撑': True,
                '说明': '图里像是已经在摆姿势，但看不出支撑。每种支撑都作备选。'}
    else:
        return {'支撑': '', '来源': '', '直接适配': False, '备选全部支撑': True,
                '说明': '看不出支撑。每种支撑都作备选。'}
    return {'支撑': img, '来源': '图片', '直接适配': False, '备选全部支撑': False, '说明': message}


def contact_from(tags, sentence):
    """One contact class from source tags. Mixed kinds stay empty. The sentence only splits anal from vaginal."""
    if not isinstance(tags, list):
        tags = []
    found = {item.strip() for item in tags if isinstance(item, str) and item.strip()}
    penetrative = '阴道性交' in found or '肛交' in found
    oral = bool(found & {'口爱', '阴茎口交', '舔阴', '舔肛', '69式'})
    foot = '足交' in found
    solo = '自慰' in found
    if sum(bool(item) for item in (penetrative, oral, foot, solo)) != 1:
        return ''
    if solo:
        return '手指或玩具接触自己的阴蒂'
    if foot:
        return '足接触阴茎'
    if '69式' in found:
        return '互相口爱'
    if oral:
        penis = '阴茎口交' in found
        vulva = '舔阴' in found
        anus = '舔肛' in found
        if penis and not vulva and not anus:
            return '嘴接触阴茎'
        if vulva and not penis and not anus:
            return '嘴接触阴部'
        return ''
    text = sentence if isinstance(sentence, str) else ''
    if '肛门' in text and '阴道' not in text:
        return '只进入肛门'
    if '阴道性交' in found:
        return '阴茎进入阴道'
    if '肛交' in found:
        return '只进入肛门'
    return ''


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
WAIST_TABLE = _by_length(tuple((phrase, phrase) for phrase in WAIST_PHRASES))
HIP_TABLE = _by_length(tuple((phrase, phrase) for phrase in HIP_PHRASES))
BUTT_TABLE = _by_length(tuple((phrase, phrase) for phrase in BUTT_PHRASES))
THIGH_TABLE = _by_length(tuple((phrase, phrase) for phrase in THIGH_PHRASES))
KNEE_TABLE = _by_length(tuple((phrase, phrase) for phrase in KNEE_PHRASES))
FOOT_TABLE = _by_length(tuple((phrase, phrase) for phrase in FOOT_PHRASES))


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


def _woman_slots(woman):
    """Closed phrases in the woman's span. A phrase that is not in the table stays empty."""
    return {
        '头': _first(woman, HEAD_CLASSES),
        '胸': _first(woman, CHEST_PHRASES),
        '腰': _first(woman, WAIST_TABLE),
        '胯': _first(woman, HIP_TABLE),
        '臀': _first(woman, BUTT_TABLE),
        '大腿': _first(woman, THIGH_TABLE),
        '膝盖': _first(woman, KNEE_TABLE),
        '脚': _first(woman, FOOT_TABLE),
    }


def _person_posture(text):
    """Library order. 跪趴 stays 扒, the same class the pose tree already uses."""
    return _first(text, POSTURE_PHRASES)


def _prompt_posture(text):
    """A person who is kneeling and bent forward is kneeling. 趴 is the torso, not the support."""
    if not text:
        return ''
    found = []
    for phrase, label in POSTURE_PHRASES:
        if phrase and phrase in text and label not in found:
            found.append(label)
    if len(found) > 1 and '扒' in found and any(label != '扒' for label in found):
        found = [label for label in found if label != '扒']
    return found[0] if found else ''


def _request_postures(text):
    """Woman and man supports from the clauses that name them. An unnamed support stays the woman's."""
    parts = re.split(r'(男人|男方|女人|女方)', text or '')
    woman = man = ''
    index = 1
    while index + 1 < len(parts):
        name, chunk = parts[index], parts[index + 1]
        posture = _prompt_posture(chunk)
        if posture == '被抱起' and '被抱' not in chunk and not re.search(r'把(女人|女方|她|男人|男方)抱', chunk):
            posture = _prompt_posture(re.sub(r'被抱起|被抱起来|被抱离地面|抱离地面', '', chunk))
            if index >= 3 and parts[index - 2].startswith('女') and name.startswith('男'):
                woman = woman or '被抱起'
            elif index >= 3 and parts[index - 2].startswith('男') and name.startswith('女'):
                man = man or '被抱起'
        if name.startswith('女') and posture and not woman:
            woman = posture
        elif name.startswith('男') and posture and not man:
            man = posture
        index += 2
    if not woman and not man:
        woman = _prompt_posture(text)
    return _correct_carried(text, woman, man)


def _correct_carried(text, woman, man):
    """被抱起写在被托起的人身上。动手抱、自己站着的人仍是站。"""
    text = text or ''
    woman_held = bool(re.search(
        r'(女人|女方).{0,48}被[^。]{0,8}抱(起|起来|离地面)'
        r'|被(男人|男方)抱离地面'
        r'|(把|将)(女人|女方|她)抱(起|起来|离地面)',
        text,
    ))
    man_held = bool(re.search(
        r'(男人|男方).{0,48}被[^。]{0,8}抱(起|起来|离地面)'
        r'|(把|将)(男人|男方)抱(起|起来|离地面)',
        text,
    ))
    if woman_held:
        woman = '被抱起'
    if man_held:
        man = '被抱起'
    elif man == '被抱起':
        man = ''
    return woman, man


def _detail_parts(slots):
    return {name: (slots or {}).get(name, '') for name, _weight in DETAIL_WEIGHTS}


def _entry_labels(sentence):
    woman, man = _person_spans(sentence)
    slots = _woman_slots(woman)
    man_slots = _woman_slots(man)
    return {
        'woman_chest': slots['胸'],
        'man_chest': _first(man, CHEST_PHRASES),
        'posture': _person_posture(woman),
        'man_posture': _person_posture(man),
        'axis': _axis(sentence),
        'head': slots['头'],
        'slots': slots,
        'man_slots': man_slots,
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
    woman_posture, man_posture = _request_postures(text)
    woman_parts, man_parts = _request_details(text)
    return {
        'woman_chest': woman_chest,
        'man_chest': man_chest,
        'posture': woman_posture,
        'man_posture': man_posture,
        'axis': axis,
        'head': _first(text, HEAD_CLASSES),
        'woman_parts': woman_parts,
        'man_parts': man_parts,
    }


def _request_details(text):
    """只把写在某人名字后面的腰、胯、臀、大腿、膝盖、脚算到这个人身上。"""
    parts = re.split(r'(男人|男方|女人|女方)', text or '')
    woman, man = [], []
    index = 1
    while index + 1 < len(parts):
        name, chunk = parts[index], parts[index + 1]
        if name.startswith('女'):
            woman.append(chunk)
        else:
            man.append(chunk)
        index += 2
    return _detail_parts(_woman_slots(''.join(woman))), _detail_parts(_woman_slots(''.join(man)))


def _stated_match(labels, entry):
    if labels['woman_chest'] and entry['woman_chest'] != labels['woman_chest']:
        return False
    if labels['man_chest'] and entry['man_chest'] != labels['man_chest']:
        return False
    if labels['axis'] and entry['axis'] != labels['axis']:
        return False
    return True


def _breakdown(query, labels, entry):
    """One line per non-empty query field. A mismatch scores 0 and does not subtract."""
    contact = query['contacts'][0] if len(query.get('contacts') or []) == 1 else ''
    parts = []

    def add(field, query_value, pose_value, weight, matched=None):
        if not query_value:
            return
        if matched is None:
            matched = query_value == pose_value
        parts.append({'field': field, 'query': query_value, 'pose': pose_value,
                      'points': weight if matched else 0})

    add('接触', contact, entry['contact'], WEIGHT_CONTACT)
    add('女方胸', labels['woman_chest'], entry['woman_chest'], WEIGHT_WOMAN_CHEST)
    add('男方胸', labels['man_chest'], entry['man_chest'], WEIGHT_MAN_CHEST)
    woman_posture = labels.get('woman_posture') or labels.get('posture') or ''
    if woman_posture:
        add('女方支撑', woman_posture, entry['posture'], WEIGHT_POSTURE,
            _posture_fits(woman_posture, entry['posture']))
    if labels.get('man_posture'):
        add('男方支撑', labels['man_posture'], entry.get('man_posture', ''), WEIGHT_POSTURE,
            _posture_fits(labels['man_posture'], entry.get('man_posture', '')))
    add('轴线', labels['axis'], entry['axis'], WEIGHT_AXIS)
    add('头', labels['head'], entry['head'], WEIGHT_HEAD)
    _add_details(add, '女方', labels.get('woman_parts'), entry.get('slots'))
    _add_details(add, '男方', labels.get('man_parts'), entry.get('man_slots'))
    return parts


def _add_details(add, side, query_parts, pose_slots):
    query_parts = query_parts or {}
    pose_slots = pose_slots or {}
    for name, weight in DETAIL_WEIGHTS:
        add(side + name, query_parts.get(name, ''), pose_slots.get(name, ''), weight)


def _score(query, labels, entry):
    entry['breakdown'] = _breakdown(query, labels, entry)
    return sum(part['points'] for part in entry['breakdown'])


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
        camera = data.get('camera') if isinstance(data.get('camera'), str) else ''
        tags = data.get('source_tags') if isinstance(data.get('source_tags'), list) else []
        source_tags = [item.strip() for item in tags if isinstance(item, str) and item.strip()]
        source_image = data.get('source_image') if isinstance(data.get('source_image'), str) else ''
        contact = contact_from(source_tags, sentence)
        number = data.get('number')
        if type(number) is not int:
            number = 10**9
        entries.append({
            'id': data['id'],
            'number': number,
            'name': data.get('name') if isinstance(data.get('name'), str) else '',
            'alias': data.get('alias') if isinstance(data.get('alias'), str) else '',
            'usable': data.get('usable_for_selection') is True and bool(sentence.strip()),
            'source_tags': source_tags,
            'source_image': source_image.strip(),
            'contact': contact,
            'contacts': [contact] if contact else [],
            'pose_sentence': sentence.strip(),
            'camera': camera.strip(),
            'breakdown': [],
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
    images = sorted(item for item in (path / IMAGE_DIR).glob('*') if item.is_file()) if (path / IMAGE_DIR).is_dir() else []
    listed = files + images
    return str([(item.name, item.stat().st_mtime_ns, item.stat().st_size) for item in listed])


def _has_image(entry):
    """True only when this entry's own source image is a file on this machine."""
    path = entry.get('source_image') or ''
    ident = entry.get('id') or ''
    if not path or not ident or any(token in ident for token in ('/', '\\', '..')):
        return False
    file = Path(path)
    return file.is_file() and file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.gif'}


def _rows(entries):
    return [{
        'id': entry['id'], 'number': entry['number'], 'name': entry['name'], 'alias': entry['alias'],
        'source_tags': entry.get('source_tags', []),
        'has_image': _has_image(entry), 'image_url': '',
        'usable': entry['usable'], 'score': entry.get('score', 0),
        'breakdown': entry.get('breakdown', []),
        'contact': entry['contact'], 'contacts': entry['contacts'],
        'woman_chest': entry['woman_chest'], 'man_chest': entry['man_chest'],
        'posture': entry['posture'], 'axis': entry['axis'], 'head': entry['head'],
        'slots': entry['slots'],
        'pose_sentence': entry['pose_sentence'], 'camera': entry['camera'],
    } for entry in entries]


def _camera_line(entry):
    camera = entry.get('camera', '').strip()
    if not camera:
        return '（这条没有镜头。不要自行补镜头。）'
    return camera


def _pose_user(original, entry):
    camera = _camera_line(entry)
    if _has_image(entry):
        instruction = (
            '已取定姿势。prompt 正文按这个格式写，前三行逐字照抄：\n'
            '图1是女人的形象。\n'
            '图2是姿势参考图。\n'
            '目标是保持图1的背景，让图1女人完成图2的姿势。\n'
            '生图详细描述：\n'
            '镜头：\n'
            '生图详细描述用连贯的话写入下面的摆法，不得改支撑、胸的朝向、站位和接触。'
            '衣着、身材、场景只取原话和图1里已经有的。\n'
            '摆法：\n' + entry['pose_sentence'] + '\n'
            '镜头素材：\n' + camera
        )
    else:
        instruction = (
            '已取定姿势。下面的摆法必须原样写入提示词，不得改支撑、站位、接触和镜头：\n'
            + entry['pose_sentence'] + '\n镜头：\n' + camera
        )
    return original.rstrip() + '\n\n' + instruction


def _pose_image(entry):
    if entry and _has_image(entry):
        return str(Path(entry['source_image']).resolve())
    return ''


def _support_hits(labels, entry):
    hits = 0
    woman = labels.get('woman_posture') or labels.get('posture') or ''
    if woman and _posture_fits(woman, entry.get('posture', '')):
        hits += 1
    man = labels.get('man_posture') or ''
    if man and _posture_fits(man, entry.get('man_posture', '')):
        hits += 1
    return hits


def _other_points(entry):
    return sum(part['points'] for part in entry.get('breakdown', [])
               if part['field'] not in ('女方支撑', '男方支撑'))


def _rank_key(item):
    points, entry = item
    return (-points, entry['number'], entry['id'])


def _choose_index(count):
    """Index inside one same-score group. Zero keeps the lowest number."""
    if count <= 1:
        return 0
    return random.randrange(count)


def _promote_tie(items, key):
    """Move one random member of the leading same-score group to the front."""
    if len(items) < 2:
        return
    best = key(items[0])
    count = 1
    while count < len(items) and key(items[count]) == best:
        count += 1
    index = _choose_index(count)
    if index:
        chosen = items.pop(index)
        items.insert(0, chosen)


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
        best = group[0][0]
        tied = [item for item in group if item[0] == best]
        pick = tied[_choose_index(len(tied))]
        if pick[1]['id'] not in seen:
            chosen.append(pick)
            seen.add(pick[1]['id'])
    unlabeled = [item for item in passed if item[1]['id'] not in seen]
    unlabeled.sort(key=_rank_key)
    if unlabeled:
        best = unlabeled[0][0]
        tied = [item for item in unlabeled if item[0] == best]
        chosen.append(tied[_choose_index(len(tied))])
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
        labels['woman_posture'] = ''
    elif choice and choice.get('支撑'):
        labels['posture'] = choice['支撑']
        labels['woman_posture'] = choice['支撑']
    if named:
        hits = [entry for entry in entries if named in (entry['name'], entry['alias'])]
        if not hits:
            return None, f'查无此条：姿势名「{named}」不在库里。已停止，不自由扩写。', []
        if len(hits) > 1:
            return None, '同名姿势超过一条，已停止，不自由扩写。', _rows(hits)
        entry = hits[0]
        if not entry['usable']:
            return None, f'查无此条：{entry["id"]} 还不是定稿。已停止，不自由扩写。', []
        if query['contacts'] and entry['contact'] not in query['contacts']:
            return None, f'查无此条：{entry["id"]} 完成不了要求的接触。已停止，不自由扩写。', _rows(hits)
        entry['score'] = _score(query, labels, entry)
        return entry, f'交合：{entry["id"]} {entry["alias"] or entry["name"]}，得分 {entry["score"]}', _rows(hits)
    if not query['contacts']:
        return None, '查无此条：没有点名，也没有可匹配的接触。已停止，不自由扩写。', []
    if len(query['contacts']) != 1:
        return None, '查无此条：一次只能匹配一种接触。已停止，不自由扩写。', []
    usable = [entry for entry in entries if entry['usable'] and entry['contact'] == query['contacts'][0]]
    if not usable:
        return None, '查无此条：没有定稿姿势能完成这个接触。已停止，不自由扩写。', []
    passed = []
    for entry in usable:
        entry['score'] = _score(query, labels, entry)
        if _stated_match(labels, entry):
            passed.append((entry['score'], entry))
    if not passed:
        return None, '查无此条：定稿姿势对不上已经写明的朝向或轴线。已停止，不自由扩写。', _rows(usable)
    if labels.get('posture') or labels.get('man_posture'):
        passed.sort(key=lambda item: (
            -_support_hits(labels, item[1]),
            -_other_points(item[1]),
            item[1]['number'],
            item[1]['id'],
        ))
        _promote_tie(passed, lambda item: (_support_hits(labels, item[1]), _other_points(item[1])))
        passed = passed[:3]
    else:
        passed = _spread_rows(passed)
    if not passed:
        return None, '查无此条：定稿姿势对不上已经写明的朝向或轴线。已停止，不自由扩写。', _rows(usable)
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
        labels = {key: query[key] for key in (
            'woman_chest', 'man_chest', 'posture', 'woman_posture', 'man_posture', 'axis', 'head',
            'woman_parts', 'man_parts')}
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
                'support': support, 'image_state': image_state, 'pose_image': ''}
    if not isinstance(pose_rules, str) or not pose_rules.strip():
        raise ValueError('交合扩写规则不能为空。')
    try:
        entries = load_library(library_dir)
    except PoseStop as error:
        return {'ok': False, 'system': '', 'user': '', 'status': str(error),
                'extraction': extraction, 'matches': '[]', 'support': support,
                'image_state': image_state, 'pose_image': ''}
    entry, status, rows = lookup(entries, query, original_request, labels, choice)
    shown = json.dumps(rows, ensure_ascii=False, indent=2)
    if entry is None:
        return {'ok': False, 'system': '', 'user': '', 'status': status,
                'extraction': extraction, 'matches': shown, 'support': support,
                'image_state': image_state, 'pose_image': ''}
    return {'ok': True, 'system': pose_rules, 'user': _pose_user(original_request, entry),
            'status': status, 'extraction': extraction, 'matches': shown, 'support': support,
            'image_state': image_state, 'pose_image': _pose_image(entry)}


def parse_pose_input(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('原话提取结果为空。')
    value = json.loads(text, object_pairs_hook=_unique)
    if isinstance(value, dict) and '接触' in value:
        return parse_extraction(text)
    return parse_contact(text)


def _zip_parts(name):
    text = str(name).replace('\\', '/').strip('/')
    parts = tuple(Path(text).parts)
    if not text or not parts or any(part in ('', '.', '..') for part in parts):
        raise ValueError('压缩包路径不合法：' + str(name))
    return parts


def _match_image(pose_id, images):
    hits = []
    for key in images:
        parts = key.split('/')
        stem = Path(parts[-1]).stem
        if stem == pose_id or (len(parts) >= 2 and parts[-2] == pose_id and stem == 'image'):
            hits.append(key)
    if len(hits) > 1:
        raise ValueError(pose_id + ' 有多张姿势图。')
    return hits[0] if hits else ''


def _verify_image(raw, filename):
    from PIL import Image
    try:
        with Image.open(io.BytesIO(raw)) as picture:
            picture.verify()
        with Image.open(io.BytesIO(raw)) as picture:
            picture.load()
            if min(picture.size) < 1:
                raise ValueError
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(filename + ' 不是可读的姿势图。') from error


def install_zip(payload, store=STORE):
    """Replace the saved pose library with the JSON and pose images inside one zip."""
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ValueError('没有读到压缩包。')
    if len(payload) > MAX_ZIP_BYTES:
        raise ValueError('压缩包超过 128MB。')
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError('这不是 zip 压缩包。') from error
    collected = []
    images = {}
    with archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
        if not files:
            raise ValueError('压缩包里没有文件。')
        if len(files) > 5000:
            raise ValueError('压缩包里的文件超过 5000 个。')
        for info in files:
            parts = _zip_parts(info.filename)
            if parts[0] == '__MACOSX' or any(part.startswith('.') for part in parts):
                continue
            filename = parts[-1]
            suffix = Path(filename).suffix.lower()
            key = '/'.join(parts)
            if suffix in IMAGE_SUFFIXES:
                if info.file_size > MAX_IMAGE_BYTES:
                    raise ValueError(filename + ' 超过 8MB。')
                if key in images:
                    raise ValueError('重复图片：' + key)
                images[key] = info
                continue
            if suffix != '.json' or not filename:
                raise ValueError('只接受姿势 JSON 和姿势图片：' + info.filename)
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
        bound = []
        used = set()
        for pose_id, item in collected:
            key = _match_image(pose_id, images)
            if key:
                if key in used:
                    raise ValueError('一张图对应了多条姿势：' + key)
                used.add(key)
            bound.append((pose_id, item, key))
        orphans = [key for key in images if key not in used]
        if orphans:
            raise ValueError('图片没有对应的姿势：' + orphans[0])
        stored = []
        for pose_id, item, key in bound:
            raw = b''
            suffix = ''
            if key:
                raw = archive.read(images[key])
                suffix = Path(key).suffix.lower()
                _verify_image(raw, Path(key).name)
            stored.append((pose_id, item, raw, suffix))
    store.mkdir(parents=True, exist_ok=True)
    image_dir = store / IMAGE_DIR
    for old in store.glob('pose_*.json'):
        old.unlink()
    if image_dir.exists():
        for old in image_dir.iterdir():
            if old.is_file():
                old.unlink()
    image_dir.mkdir(parents=True, exist_ok=True)
    for pose_id, item, raw, suffix in stored:
        if raw:
            path = image_dir / f'{pose_id}{suffix}'
            path.write_bytes(raw)
            item['source_image'] = str(path.resolve())
        filename = pose_id if pose_id.startswith('pose_') else 'pose_' + pose_id
        (store / f'{filename}.json').write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return len(stored)


def pack_library(library_dir, destination):
    """Write a zip of pose JSON plus each entry's own pose image."""
    library = Path(library_dir)
    files = sorted(item for item in library.glob('pose_*.json') if item.is_file())
    if not files:
        raise ValueError('姿势库目录里没有条目：' + str(library))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_count = 0
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            data = json.loads(file.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or not isinstance(data.get('id'), str) or not data['id'].strip():
                raise ValueError(file.name + ' 缺少 id。')
            pose_id = data['id'].strip()
            source = data.get('source_image') if isinstance(data.get('source_image'), str) else ''
            image = Path(source) if source else None
            if image and image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                arcname = pose_id + image.suffix.lower()
                data['source_image'] = arcname
                archive.write(image, arcname=arcname)
                image_count += 1
            archive.writestr(pose_id + '.json', json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return len(files), image_count

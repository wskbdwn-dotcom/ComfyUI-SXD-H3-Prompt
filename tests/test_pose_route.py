"""Pose lookup stays deterministic and the new workflow wires it in front of expansion."""
import io
import json
from pathlib import Path
from PIL import Image
import sys
import tempfile
import unittest
import zipfile

import test_image_tasks as image_tests
import test_v2 as legacy

route = sys.modules[legacy.spec.name + '.pose_route']
nodes = sys.modules[legacy.spec.name + '.image_nodes']
ROOT = Path(nodes.__file__).parent
REAL_LIBRARY = Path('/Users/mt/gpt/生图研究/姿势库/条目')
POSE = {'route': 'pose', 'contacts': ['足接触阴茎'], 'named_pose': '',
        'people': 2, 'vertical': True, 'face_visible': True}


def dump(value):
    return json.dumps(value, ensure_ascii=False)


class ContactParseTests(unittest.TestCase):
    def test_general_rejects_extra_fields(self):
        self.assertEqual(route.parse_contact('{"route":"general"}')['route'], 'general')
        with self.assertRaises(ValueError):
            route.parse_contact('{"route":"general","contacts":[]}')

    def test_pose_record_keeps_only_known_labels(self):
        parsed = route.parse_contact(dump(POSE))
        self.assertEqual(parsed['contacts'], ['足接触阴茎'])
        with self.assertRaises(ValueError):
            bad = dict(POSE, contacts=['亲吻'])
            route.parse_contact(dump(bad))


class ContactNodeTests(unittest.TestCase):
    setUp = image_tests.ImageTaskTests.setUp
    tearDown = image_tests.ImageTaskTests.tearDown
    config = image_tests.ImageTaskTests.config
    run_task = image_tests.ImageTaskTests.run_task

    def test_contact_task_returns_closed_json(self):
        text, _ = self.run_task(answer=dump(POSE), task='contact')
        self.assertEqual(json.loads(text)['contacts'], ['足接触阴茎'])

    def test_contact_task_requires_a_reference_image(self):
        with self.assertRaises(ValueError):
            self.run_task(answer='{"route":"general"}', task='contact', image=None)


class LibraryTests(unittest.TestCase):
    def write(self, folder, rows):
        for row in rows:
            (folder / f'{row["id"]}.json').write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')

    def ask(self, library, record=POSE, rules='POSE RULES 不要另选姿势', request='保持衣服'):
        return route.prepare(dump(record), request, 'RULES', rules, library)

    def test_general_route_does_not_read_the_library(self):
        result = route.prepare('{"route":"general"}', '把外套改成蓝色', 'RULES', '', '/missing/poses')
        self.assertEqual((result['system'], result['user'], result['matches']), ('RULES', '把外套改成蓝色', '[]'))
        self.assertIn('不交合', result['status'])
        self.assertIn('"route": "general"', result['extraction'])

    def test_one_usable_entry_is_locked_into_the_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), [{
                'id': 'pose_test', 'name': '测试', 'alias': '双脚夹住',
                'usable_for_selection': True, 'source_tags': ['足交'],
                'pose_sentence': '女人头朝前，胸背对男人，腰直，胯朝前，臀贴着男人，大腿伸直，膝盖伸直，双脚着地，站立。足接触阴茎。',
                'camera': '镜头在两人身体的一侧。',
            }])
            result = self.ask(tmp)
        self.assertTrue(result['ok'])
        self.assertIn('不要另选姿势', result['system'])
        self.assertIn('不得改支撑', result['user'])
        self.assertIn('双脚着地，站立。足接触阴茎。', result['user'])
        self.assertIn('镜头：\n镜头在两人身体的一侧。', result['user'])
        row = json.loads(result['matches'])[0]
        self.assertEqual(row['camera'], '镜头在两人身体的一侧。')
        self.assertEqual(row['source_tags'], ['足交'])
        self.assertEqual(row['contact'], '足接触阴茎')
        self.assertEqual(row['has_image'], False)
        self.assertEqual(row['image_url'], '')
        self.assertEqual(row['slots']['头'], '头朝前')
        self.assertEqual(row['slots']['胸'], '背对对方')
        self.assertEqual(row['slots']['腰'], '腰直')
        self.assertEqual(row['slots']['胯'], '胯朝前')
        self.assertEqual(row['slots']['臀'], '')
        self.assertEqual(row['slots']['大腿'], '大腿伸直')
        self.assertEqual(row['slots']['膝盖'], '膝盖伸直')
        self.assertEqual(row['slots']['脚'], '脚着地')
        self.assertIn({'field': '接触', 'query': '足接触阴茎', 'pose': '足接触阴茎', 'points': 60}, row['breakdown'])
        self.assertIn('保持衣服', result['user'])
        self.assertIn('pose_test', result['status'])
        self.assertIn('pose_test', result['matches'])

    def test_contact_comes_from_source_tags(self):
        self.assertEqual(route.contact_from(['肛交', '阴道性交'], '阴茎进入阴道。'), '阴茎进入阴道')
        self.assertEqual(route.contact_from(['阴道性交', '肛交'], '只进入肛门。'), '只进入肛门')
        self.assertEqual(route.contact_from(['肛交'], ''), '只进入肛门')
        self.assertEqual(route.contact_from(['足交'], ''), '足接触阴茎')
        self.assertEqual(route.contact_from(['69式', '口爱', '舔阴', '阴茎口交'], ''), '互相口爱')
        self.assertEqual(route.contact_from(['口爱', '阴茎口交'], ''), '嘴接触阴茎')
        self.assertEqual(route.contact_from(['口爱', '舔阴'], ''), '嘴接触阴部')
        self.assertEqual(route.contact_from(['口爱', '舔阴', '阴茎口交'], ''), '')
        self.assertEqual(route.contact_from(['自慰', '阴道性交'], ''), '')
        self.assertEqual(route.contact_from(None, '阴茎进入阴道。'), '')
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), [{
                'id': 'pose_sentence_only', 'usable_for_selection': True,
                'pose_sentence': '阴茎进入阴道。',
            }])
            missed = self.ask(tmp, dict(POSE, contacts=['阴茎进入阴道']))
        self.assertFalse(missed['ok'])
        self.assertIn('没有定稿姿势', missed['status'])

    def test_unusable_and_missing_entries_stop_but_keep_the_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), [{
                'id': 'pose_closed', 'alias': '还不能用', 'usable_for_selection': False,
                'pose_sentence': '足接触阴茎。',
            }])
            missed = self.ask(tmp)
            self.assertFalse(missed['ok'])
            self.assertIn('已停止', missed['status'])
            self.assertEqual(missed['matches'], '[]')
            named = self.ask(tmp, dict(POSE, named_pose='还不能用'))
            self.assertFalse(named['ok'])
            self.assertIn('还不是定稿', named['status'])

    def test_finalized_entries_rank_by_score_without_a_manual_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), [
                {'id': 'pose_a', 'number': 8, 'usable_for_selection': True, 'source_tags': ['足交'],
                 'pose_sentence': '女人头转向他。足接触阴茎。', 'image_check': {'status': 'conflict'}},
                {'id': 'pose_b', 'number': 3, 'usable_for_selection': True, 'source_tags': ['足交'],
                 'pose_sentence': '另一条。足接触阴茎。'},
            ])
            numbered = self.ask(tmp)
            self.assertTrue(numbered['ok'])
            self.assertTrue('pose_a' in numbered['status'] or 'pose_b' in numbered['status'])
            self.assertIn('（这条没有镜头。不要自行补镜头。）', numbered['user'])
            headed = self.ask(tmp, request='头转向他，保持衣服')
            self.assertTrue(headed['ok'])
            self.assertIn('pose_a', headed['status'])
            self.assertIn('得分 75', headed['status'])

    def test_installed_library_selects_a_finalized_pose(self):
        if not REAL_LIBRARY.is_dir():
            self.skipTest('local pose library is absent')
        entries = route.load_library(REAL_LIBRARY)
        self.assertGreaterEqual(len(entries), 573)
        self.assertGreater(sum(entry['usable'] for entry in entries), 0)
        result = self.ask(REAL_LIBRARY)
        self.assertTrue(result['ok'])
        self.assertIn('得分', result['status'])

    def test_extraction_json_and_image_support_choose_the_pose(self):
        document = {
            '人数': 2, 'alias': '', '接触': '足接触阴茎', '支撑': '', '轴线': '',
            'body_parts': {
                '女方': {'头': '头转向他', '胸': '', '腰': '', '胯': '', '臀': '', '大腿': '', '膝盖': '', '脚': ''},
                '男方': {'头': '', '胸': '', '腰': '', '胯': '', '臀': '', '大腿': '', '膝盖': '', '脚': ''},
            },
        }
        image = {'支撑': '跪', '姿势已准备': False, '事实': ['女人跪在画面里。']}
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), [
                {'id': 'pose_stand', 'number': 1, 'usable_for_selection': True, 'source_tags': ['足交'],
                 'pose_sentence': '女人站着。足接触阴茎。'},
                {'id': 'pose_kneel', 'number': 9, 'usable_for_selection': True, 'source_tags': ['足交'],
                 'pose_sentence': '女人头转向他，跪着。足接触阴茎。',
                 'image_check': {'status': 'conflict'}},
            ])
            result = route.prepare(dump(document), '保持衣服', 'RULES', 'POSE RULES 不要另选姿势', tmp, dump(image))
        self.assertTrue(result['ok'])
        shown = json.loads(result['extraction'])
        self.assertEqual(shown['接触'], '足接触阴茎')
        self.assertEqual(shown['body_parts']['女方']['头'], '头转向他')
        self.assertIn('按图里的支撑优先', json.loads(result['support'])['说明'])
        self.assertEqual(json.loads(result['image_state'])['事实'], ['女人跪在画面里。'])
        self.assertIn('pose_kneel', result['status'])

    def test_stated_standing_and_forward_bend_outrank_a_kneeling_man(self):
        document = {
            '人数': 2, 'alias': '', '接触': '阴茎进入阴道', '支撑': '站',
            '女方支撑': '站', '男方支撑': '', '轴线': '男方在女人身后',
            'body_parts': {
                '女方': {'头': '', '胸': '', '腰': '腰后仰', '胯': '', '臀': '臀抬起', '大腿': '', '膝盖': '', '脚': ''},
                '男方': {'头': '', '胸': '', '腰': '', '胯': '', '臀': '', '大腿': '', '膝盖': '', '脚': ''},
            },
        }
        request = '女人站立弯腰翘起臀部，男人站立从后面插入女人阴道'
        with tempfile.TemporaryDirectory() as tmp:
            self.write(Path(tmp), [
                {'id': 'pose_kneel_man', 'number': 140, 'usable_for_selection': True,
                 'source_tags': ['阴道性交'],
                 'pose_sentence': '女人站着，腰直，臀抬起，背对男人。男人在女人身后跪下。阴茎进入阴道。'},
                {'id': 'pose_stand_both', 'number': 200, 'usable_for_selection': True,
                 'source_tags': ['阴道性交'],
                 'pose_sentence': '女人站着，腰前折，臀抬起，背对男人。男人在女人身后站着。阴茎进入阴道。'},
            ])
            result = route.prepare(
                dump(document), request, 'RULES', 'POSE RULES 不要另选姿势', tmp, '')
        shown = json.loads(result['extraction'])
        self.assertEqual(shown['男方支撑'], '站')
        self.assertEqual(shown['女方支撑'], '站')
        self.assertEqual(shown['body_parts']['女方']['腰'], '腰前折')
        self.assertEqual(shown['body_parts']['女方']['臀'], '臀抬起')
        self.assertEqual(shown['body_parts']['女方']['胸'], '背对对方')
        self.assertIn('pose_stand_both', result['status'])

    def test_a_one_item_slot_list_is_read_as_the_phrase(self):
        document = {
            '人数': 2, 'alias': '', '接触': '阴茎进入阴道', '支撑': '站', '轴线': '',
            'body_parts': {
                '女方': {'头': '', '胸': '', '腰': '', '胯': '', '臀': '', '大腿': ['大腿抬起'], '膝盖': '', '脚': ''},
                '男方': {'头': '', '胸': '', '腰': '', '胯': '', '臀': '', '大腿': '', '膝盖': '', '脚': ''},
            },
        }
        parsed = route.parse_extraction(dump(document))
        self.assertEqual(parsed['document']['body_parts']['女方']['大腿'], '大腿抬起')
        self.assertEqual(parsed['document']['支撑'], '站')

    def test_zip_keeps_pose_images_and_rejects_unmatched_ones(self):
        def packed(files):
            blob = io.BytesIO()
            with zipfile.ZipFile(blob, 'w') as archive:
                for name, text in files.items():
                    archive.writestr(name, text)
            return blob.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            first = json.dumps([{'id': 'pose_one', 'pose_sentence': '足接触阴茎。'}])
            self.assertEqual(route.install_zip(packed({'list.json': first, 'pose_one.png': self.picture()}), store), 1)
            saved = json.loads((store / 'pose_one.json').read_text())
            self.assertTrue(Path(saved['source_image']).is_file())
            self.assertEqual(Path(saved['source_image']).name, 'pose_one.png')
            second = json.dumps({'id': 'pose_two', 'pose_sentence': '嘴接触阴茎。'})
            self.assertEqual(route.install_zip(packed({'again.json': second}), store), 1)
            self.assertFalse((store / 'pose_one.json').exists())
            self.assertFalse((store / 'images' / 'pose_one.png').exists())
            self.assertTrue((store / 'pose_two.json').is_file())
            with self.assertRaises(ValueError):
                route.install_zip(packed({'picture.png': self.picture()}), store)
            self.assertTrue((store / 'pose_two.json').is_file())
            with self.assertRaises(ValueError):
                route.install_zip(packed({
                    'pose_two.json': second,
                    'picture.png': self.picture(),
                }), store)

    def picture(self):
        blob = io.BytesIO()
        Image.new('RGB', (4, 4), (10, 20, 30)).save(blob, format='PNG')
        return blob.getvalue()

    def test_selected_pose_image_is_named_as_reference_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            picture = folder / 'pose.png'
            Image.new('RGB', (4, 4), (1, 2, 3)).save(picture)
            self.write(folder, [{
                'id': 'pose_kneel', 'number': 1, 'usable_for_selection': True, 'source_tags': ['足交'],
                'pose_sentence': '女人跪着。足接触阴茎。', 'source_image': str(picture),
            }])
            result = self.ask(folder)
        self.assertTrue(result['ok'])
        self.assertIn('图1是女人的形象。', result['user'])
        self.assertIn('图2是姿势参考图。', result['user'])
        self.assertIn('生图详细描述：', result['user'])
        self.assertIn('女人跪着。足接触阴茎。', result['user'])
        self.assertEqual(result['pose_image'], str(picture.resolve()))


class PoseWorkflowTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'workflows/qwen'
        self.workflow = json.loads((folder / 'image2image_qwen2.1-gemma-v9-pose-route.json').read_text())
        self.nodes = {node['id']: node for node in self.workflow['nodes']}
        self.links = {link[0]: link for link in self.workflow['links']}
        self.rules = (folder / 'contact-extract-rules.txt').read_text()

    def test_contact_lookup_sits_in_front_of_the_existing_expander(self):
        contact = self.nodes[1402]
        lookup = self.nodes[1403]
        expand = self.nodes[1151]
        self.assertEqual(contact['type'], 'SXDGemmaImageTask')
        self.assertEqual(contact['widgets_values'][0], 'extract')
        self.assertEqual(self.nodes[1408]['widgets_values'][0], 'image_support')
        self.assertEqual(self.nodes[1407]['widgets_values'][0],
                         (ROOT / 'workflows/qwen/image-support-rules.txt').read_text())
        self.assertIn('脚被裁掉也是站', self.nodes[1407]['widgets_values'][0])
        self.assertEqual(lookup['type'], 'SXDPoseRoute')
        self.assertEqual(expand['widgets_values'][0], 'prompt')
        self.assertEqual(self.nodes[1400]['widgets_values'][0], self.rules)
        self.assertEqual(self.nodes[1401]['widgets_values'][0], route.POSE_RULES)
        self.assertEqual(self.nodes[1403]['inputs'][3]['name'], 'pose_rules')
        self.assertEqual(self.links[126][1:5], [1402, 0, 1403, 0])
        self.assertEqual(self.links[130][1:5], [1403, 3, 1405, 0])
        self.assertEqual(self.links[137][1:5], [1403, 5, 1410, 0])
        self.assertEqual(self.nodes[1403]['inputs'][4]['name'], 'image_json')
        self.assertEqual(self.links[131][1:5], [1403, 4, 1406, 0])
        self.assertTrue(self._via_if('读取图片支撑', '读图开关 · 开则识别支撑', '姿势查库 · 取一条或停止', 4))
        self.assertTrue(self._via_if('读取图片支撑', '读图开关 · 开则识别支撑', '图里的事实', 0))
        self.assertTrue(self._via_set('姿势查库 · 取一条或停止', 0, '查库系统提示词', '自定义规则 · Gemma 直接扩写', 1))
        self.assertTrue(self._via_set('姿势查库 · 取一条或停止', 1, '查库用户提示词', '自定义规则 · Gemma 直接扩写', 2))
        self.assertEqual(self.nodes[1403]['outputs'][7]['name'], 'pose_image')
        self.assertTrue(self._via_set('姿势查库 · 取一条或停止', 7, '姿势图', '原 Qwen 2.1 编码 · 接收增强提示词', 2))
        self.assertTrue(self._via_set('姿势查库 · 取一条或停止', 7, '姿势图', '自定义规则 · Gemma 直接扩写', 5))
        self.assertEqual(self.nodes[1412]['type'], 'PreviewImage')
        self.assertEqual(self.nodes[1412]['title'], '选中的姿势图')
        self.assertEqual(self.links[140][1:5], [1403, 7, 1412, 0])
        self.assertEqual(self.links[139][1:5], [1337, 0, 1403, 5])
        self.assertEqual(self._widget('读图识别支撑'), True)
        self.assertEqual(self._widget('畸形检查'), True)
        self.assertIn('"enable_thinking": false', self.nodes[1150]['widgets_values'][0])
        self.assertEqual(self.nodes[1127]['widgets_values'][0], 840)
        self.assertEqual(self._cross_group_links(), [])
        self.assertEqual(self.nodes[1403]['inputs'][5]['name'], 'image2')
        self.assertNotIn(31, self.links)
        self.assertNotIn(32, self.links)
        self.assertIn('不要另选姿势', route.POSE_RULES)
        self.assertIn('图1是女人的形象。', route.POSE_RULES)
        self.assertIn('保持图1的背景', route.POSE_RULES)

    def _by_title(self, title):
        found = [node for node in self.workflow['nodes'] if node.get('title') == title]
        self.assertEqual(len(found), 1, title)
        return found[0]

    def _widget(self, title):
        return self._by_title(title)['widgets_values'][0]

    def _via_set(self, src_title, src_slot, name, dst_title, dst_slot):
        setter = next(node for node in self.workflow['nodes']
                      if node['type'] == 'SetNode' and node['widgets_values'][0] == name
                      and self.links[node['inputs'][0]['link']][1:3] == [self._by_title(src_title)['id'], src_slot])
        getter_ids = {node['id'] for node in self.workflow['nodes']
                      if node['type'] == 'GetNode' and node['widgets_values'][0] == name}
        dst = self._by_title(dst_title)
        origin = self.links[dst['inputs'][dst_slot]['link']][1]
        return setter is not None and origin in getter_ids

    def _via_if(self, src_title, switch_title, dst_title, dst_slot):
        switch = self._by_title(switch_title)
        src = self._by_title(src_title)
        dst = self._by_title(dst_title)
        true_from = self.links[switch['inputs'][1]['link']][1]
        out_to = self.links[dst['inputs'][dst_slot]['link']][1]
        return true_from == src['id'] and out_to == switch['id']

    def _cross_group_links(self):
        def group_of(node):
            x, y = node['pos']
            hits = []
            for group in self.workflow['groups']:
                left, top, width, height = group['bounding']
                if left <= x <= left + width and top <= y <= top + height:
                    hits.append(group['title'])
            return hits
        crossed = []
        for link in self.workflow['links']:
            if group_of(self.nodes[link[1]]) != group_of(self.nodes[link[3]]):
                crossed.append(link[0])
        return crossed


class ApiWorkflowTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'workflows/qwen'
        self.workflow = json.loads((folder / 'image2image_qwen2.1-qwen38-api-pose-route.json').read_text())
        self.nodes = {node['id']: node for node in self.workflow['nodes']}
        self.links = {link[0]: link for link in self.workflow['links']}

    def test_remote_api_replaces_gemma_and_keeps_the_full_text(self):
        self.assertEqual(self._by_title('API 模型名')['widgets_values'], ['qwen3.8'])
        self.assertEqual(self._by_title('温度')['widgets_values'], [1.0])
        self.assertEqual(self._by_title('最大生成长度')['widgets_values'], [8192])
        self.assertEqual(self._by_title('Top P')['widgets_values'], [0.95])
        self.assertIn('"enable_thinking": false', self._by_title('思考强度 · 改这个 JSON')['widgets_values'][0])
        client = self._by_title('远程 API 地址和密钥 · 调用时传入')
        self.assertEqual(client['type'], 'SXDOpenAIClient')
        self.assertEqual(client['widgets_values'][0], 'http://127.0.0.1:8000/v1')
        self.assertEqual(client['widgets_values'][1], '')
        calls = [node for node in self.workflow['nodes'] if node['type'] == 'OAIAPI_ChatCompletion']
        self.assertEqual(len(calls), 4)
        self.assertFalse(any(node['type'] in ('SXDGemmaImageTask', 'H3PromptLiveDisplay')
                             for node in self.workflow['nodes']))
        self.assertTrue(PoseWorkflowTests._via_set(self, 'API 模型名', 0, 'API模型', '提取原话 JSON · API', 1))
        chain = []
        node = self._by_title('Set_API选项')
        while node['type'] != 'OAIAPI_ExtraBody':
            link_id = next(item['link'] for item in node['inputs'] if item['type'] == 'OAIAPI_OPTIONS')
            node = self.nodes[self.links[link_id][1]]
            chain.append(node['type'])
        self.assertEqual(chain, ['OAIAPI_Seed', 'OAIAPI_MaxTokens', 'OAIAPI_TopP',
                                 'OAIAPI_Temperature', 'OAIAPI_ExtraBody'])
        self.assertEqual(self.links[self._by_title('提取原话 · 完整返回')['inputs'][0]['link']][1],
                         self._by_title('提取原话 JSON · API')['id'])
        self.assertEqual(self.links[self._by_title('扩写完整返回 · 只有 prompt 字段')['inputs'][0]['link']][1],
                         self._by_title('自定义规则 · API 直接扩写')['id'])
        self.assertEqual(self.links[self._by_title('检查完整返回')['inputs'][0]['link']][1],
                         self._by_title('大结构检查一次 · API')['id'])
        self.assertIsNone(self._by_title('提取原话 JSON · API')['inputs'][7]['link'])
        self.assertIsNotNone(self._by_title('读取图片支撑 · API')['inputs'][7]['link'])
        for title, task, show in (
                ('提取原话 · 校验结果', 'extract', '提取原话 · 完整返回'),
                ('读图支撑 · 校验结果', 'image_support', '读图支撑 · 完整返回'),
                ('扩写 · 校验结果', 'prompt', '扩写完整返回 · 只有 prompt 字段'),
                ('检查 · 校验结果', 'anatomy', '检查完整返回')):
            node = self._by_title(title)
            self.assertEqual(node['type'], 'SXDImageTaskResult')
            self.assertEqual(node['widgets_values'], [task])
            self.assertEqual(self.links[node['inputs'][0]['link']][1], self._by_title(show)['id'])
        self.assertEqual(self.links[self._by_title('姿势查库 · 取一条或停止')['inputs'][0]['link']][1],
                         self._by_title('提取原话 · 校验结果')['id'])
        self.assertEqual(self.links[self._by_title('读图开关 · 开则识别支撑')['inputs'][1]['link']][1],
                         self._by_title('读图支撑 · 校验结果')['id'])
        self.assertEqual(nodes.SXDImageTaskResult().run('```json\n{"prompt":" 保持衣服 "}\n```', 'prompt'),
                         ('{"prompt":"保持衣服"}', '```json\n{"prompt":" 保持衣服 "}\n```'))
        with self.assertRaises(ValueError) as raised:
            nodes.SXDImageTaskResult().run('不是 JSON', 'extract')
        self.assertIn('不是 JSON', str(raised.exception))
        self.assertEqual(PoseWorkflowTests._cross_group_links(self), [])

    def test_key_comes_from_the_call(self):
        import sys
        from unittest import mock
        openai = mock.Mock()
        with mock.patch.dict(sys.modules, {'openai': openai}):
            nodes.SXDOpenAIClient().run('http://example/v1', 'call-key', 1, 30)
            self.assertEqual(openai.OpenAI.call_args.kwargs['api_key'], 'call-key')
            nodes.SXDOpenAIClient().run('http://example/v1', '  ', 0, 10)
            self.assertEqual(openai.OpenAI.call_args.kwargs['api_key'], '-')

    def _by_title(self, title):
        return PoseWorkflowTests._by_title(self, title)

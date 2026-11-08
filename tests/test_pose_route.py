"""Pose lookup stays deterministic and the new workflow wires it in front of expansion."""
import io
import json
from pathlib import Path
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
                'usable_for_selection': True, 'pose_sentence': '摆法已写明。足接触阴茎。',
            }])
            result = self.ask(tmp)
        self.assertTrue(result['ok'])
        self.assertIn('不要另选姿势', result['system'])
        self.assertIn('不得改支撑', result['user'])
        self.assertIn('摆法已写明。足接触阴茎。', result['user'])
        self.assertIn('保持衣服', result['user'])
        self.assertIn('pose_test', result['status'])
        self.assertIn('pose_test', result['matches'])

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
                {'id': 'pose_a', 'number': 8, 'usable_for_selection': True,
                 'pose_sentence': '女人头转向他。足接触阴茎。', 'image_check': {'status': 'conflict'}},
                {'id': 'pose_b', 'number': 3, 'usable_for_selection': True,
                 'pose_sentence': '另一条。足接触阴茎。'},
            ])
            numbered = self.ask(tmp)
            self.assertTrue(numbered['ok'])
            self.assertIn('pose_b', numbered['status'])
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
                {'id': 'pose_stand', 'number': 1, 'usable_for_selection': True,
                 'pose_sentence': '女人站着。足接触阴茎。'},
                {'id': 'pose_kneel', 'number': 9, 'usable_for_selection': True,
                 'pose_sentence': '女人头转向他，跪着。足接触阴茎。',
                 'image_check': {'status': 'conflict'}},
            ])
            result = route.prepare(dump(document), '保持衣服', 'RULES', 'POSE RULES 不要另选姿势', tmp, dump(image))
        self.assertTrue(result['ok'])
        shown = json.loads(result['extraction'])
        self.assertEqual(shown['接触'], '足接触阴茎')
        self.assertEqual(shown['body_parts']['女方']['头'], '头转向他')
        self.assertIn('按图里的支撑选', json.loads(result['support'])['说明'])
        self.assertEqual(json.loads(result['image_state'])['事实'], ['女人跪在画面里。'])
        self.assertIn('pose_kneel', result['status'])

    def test_zip_replaces_saved_json_and_rejects_images(self):
        def packed(files):
            blob = io.BytesIO()
            with zipfile.ZipFile(blob, 'w') as archive:
                for name, text in files.items():
                    archive.writestr(name, text)
            return blob.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            first = json.dumps([{'id': 'pose_one', 'pose_sentence': '足接触阴茎。'}])
            self.assertEqual(route.install_zip(packed({'list.json': first}), store), 1)
            self.assertTrue((store / 'pose_one.json').is_file())
            second = json.dumps({'id': 'pose_two', 'pose_sentence': '嘴接触阴茎。'})
            self.assertEqual(route.install_zip(packed({'again.json': second}), store), 1)
            self.assertFalse((store / 'pose_one.json').exists())
            self.assertTrue((store / 'pose_two.json').is_file())
            with self.assertRaises(ValueError):
                route.install_zip(packed({'picture.png': 'not-an-image'}), store)
            self.assertTrue((store / 'pose_two.json').is_file())


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
        self.assertEqual(lookup['type'], 'SXDPoseRoute')
        self.assertEqual(expand['widgets_values'][0], 'prompt')
        self.assertEqual(self.nodes[1400]['widgets_values'][0], self.rules)
        self.assertEqual(self.nodes[1401]['widgets_values'][0], route.POSE_RULES)
        self.assertEqual(self.nodes[1403]['inputs'][3]['name'], 'pose_rules')
        self.assertEqual(self.links[126][1:5], [1402, 0, 1403, 0])
        self.assertEqual(self.links[130][1:5], [1403, 3, 1405, 0])
        self.assertEqual(self.links[136][1:5], [1408, 0, 1409, 0])
        self.assertEqual(self.links[137][1:5], [1403, 5, 1410, 0])
        self.assertEqual(self.links[138][1:5], [1408, 0, 1403, 4])
        self.assertEqual(self.nodes[1403]['inputs'][4]['name'], 'image_json')
        self.assertEqual(self.links[131][1:5], [1403, 4, 1406, 0])
        self.assertEqual(self.links[116][1:5], [1403, 0, 1151, 1])
        self.assertEqual(self.links[129][1:5], [1403, 1, 1151, 2])
        self.assertNotIn(31, self.links)
        self.assertNotIn(32, self.links)
        self.assertIn('不要另选姿势', route.POSE_RULES)

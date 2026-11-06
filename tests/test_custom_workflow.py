"""The editable workflow sends rules directly to the existing prompt node."""
import json
from pathlib import Path
import unittest
from unittest import mock

import test_image_tasks as image_tests
import test_spatial_plan as spatial_tests
import test_v2 as legacy

ROOT = Path(image_tests.nodes.__file__).parent


class CustomWorkflowTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'workflows/qwen'
        self.text = (folder / 'image2image_qwen2.1-gemma-v9-live-custom-rules.json').read_text()
        self.workflow = json.loads(self.text)
        self.nodes = {n['id']: n for n in self.workflow['nodes']}
        self.links = {l[0]: l for l in self.workflow['links']}
        self.baseline = json.loads((folder / 'image2image_qwen2.1-gemma-v9-live-structured.json').read_text())

    def test_all_links_are_bidirectional(self):
        spatial_tests.PublishedWorkflowTests.test_all_links_are_bidirectional(self)

    def test_editable_rules_feed_direct_prompt_and_live_outputs(self):
        node = self.nodes[1151]
        self.assertEqual(node['type'], 'SXDGemmaImageTask')
        self.assertEqual(node['widgets_values'], ['prompt', False])
        self.assertEqual(node['inputs'][1]['name'], 'system_prompt_override')
        self.assertEqual(self.links[31][1:5], [1201, 0, 1151, 1])
        self.assertEqual(self.links[32][1:5], [1200, 0, 1151, 2])
        self.assertEqual(self.links[42][1:5], [1151, 0, 1213, 0])
        self.assertEqual(self.links[34][1:5], [1213, 0, 1204, 0])
        self.assertEqual(self.links[113][1:5], [1151, 1, 1358, 0])
        self.assertEqual(self.links[114][1:5], [1152, 1, 1359, 0])
        self.assertFalse(any(n['type'] in ('SXDGemmaSpatialPlan', 'SXDSpatialPlanAssemble')
                             for n in self.nodes.values()))

    def test_rules_are_present_and_public_template_has_no_private_inputs(self):
        rules = (ROOT / 'workflows/qwen/custom-image-edit-rules.txt').read_text().strip()
        self.assertEqual(self.nodes[1201]['widgets_values'][0], rules)
        self.assertIn('Simplified Chinese', rules)
        self.assertIn('exactly one nonempty string field named "prompt"', rules)
        self.assertEqual(self.nodes[1200]['widgets_values'][0], '')
        self.assertEqual(self.nodes[1109]['widgets_values'][0], '')
        self.assertEqual(self.nodes[1109]['widgets_values_named']['string'], '')
        self.assertNotIn('r2.spellpic.com', self.text)
        self.assertNotIn('腰间水壶', self.text)

    def test_generation_settings_and_unrelated_nodes_links_are_unchanged(self):
        modified = {1151, 1200, 1201, 1213, 1358}
        removed = {1360, 1361}
        for node in self.baseline['nodes']:
            if node['id'] not in modified | removed:
                self.assertEqual(self.nodes[node['id']], node)
        for link in self.baseline['links']:
            if link[0] not in (42, 115, 116, 117):
                self.assertEqual(self.links[link[0]], link)
        self.assertEqual(self.workflow['definitions'], self.baseline['definitions'])


class CustomRulesRuntimeTests(unittest.TestCase):
    setUp = legacy.InferenceAdapterTests.setUp
    tearDown = legacy.InferenceAdapterTests.tearDown
    config = image_tests.ImageTaskTests.config

    def test_user_rules_reach_model_verbatim_without_structured_validation(self):
        self.response = {'choices': [{'finish_reason': 'stop',
                                     'message': {'content': '{"prompt":"将外套改为蓝色。"}'}}]}
        with mock.patch.object(spatial_tests.plan_module, 'normalized_plan',
                               side_effect=AssertionError('Must not invoke spatial planner')):
            for rules in ('只修改服装颜色，输出中文 prompt JSON。', '按用户自定义的新规则生成 prompt JSON。'):
                output = image_tests.nodes.SXDGemmaImageTask().run(
                    config_override=self.config(), system_prompt_override=rules,
                    user_prompt='把外套改成蓝色', seed=42, task='prompt', image=self.args['first_frame'])
                self.assertEqual(self.call['messages'][0]['content'], rules)
                self.assertEqual(json.loads(output[0]), {'prompt': '将外套改为蓝色。'})
        self.assertEqual(self.events.count('generate'), 2)


if __name__ == '__main__':
    unittest.main()

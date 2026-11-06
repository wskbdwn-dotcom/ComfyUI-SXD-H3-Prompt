"""Contract failures block assembly; model integration uses the shared fake backend."""
import json
from pathlib import Path
import sys
import unittest

import test_v2 as legacy
import test_image_tasks as image_tests

nodes = sys.modules[legacy.spec.name + '.image_nodes']
plan_module = sys.modules[legacy.spec.name + '.spatial_plan']


def fixture():
    return {
        'version': 1, 'scope': 'spatial', 'goal': '让图中人物接过另一人递来的书',
        'people': [
            dict(id='p1', label='戴眼镜的人物', origin='reference', position='画面左侧',
                 posture='visible_only', posture_detail='上身直立，保留原图可见的支撑关系',
                 torso_orientation='朝向画面右侧', action='伸手接住同一本书'),
            dict(id='p2', label='递书人物', origin='added', position='画面右侧',
                 posture='standing', posture_detail='上身略向左前倾',
                 torso_orientation='朝向画面左侧的人物', action='伸手递出同一本书'),
        ],
        'objects': [dict(id='o1', name='书', owner_id='p2', mount='hand', location='两人之间')],
        'contacts': [dict(source=dict(kind='object', id='o1', part='侧缘'),
                          target=dict(kind='person', id='p1', part='手掌'), relation='接触',
                          spatial_detail='双方的手接触同一本书，书本位于两人之间的空隙')],
        'camera': dict(mode='adjust', direction='从两人正前方观察',
                       height='高度与戴眼镜人物的肩部齐平',
                       framing='取景包含两人头顶至腰部以及交接的书',
                       visibility='两人横向分开，双方的手和书本从躯干之间的空隙露出'),
        'retain': '延续参考图中的人物身份、服装与画风',
    }


class PlanContractTests(unittest.TestCase):
    def encoded(self, value=None):
        return json.dumps(fixture() if value is None else value, ensure_ascii=False)

    def test_complete_plan_compiles_posture_and_camera_into_prompt_only_json(self):
        external, prompt = plan_module.assemble(self.encoded())
        self.assertEqual(json.loads(external), {'prompt': prompt})
        self.assertIn('递书人物位于画面右侧，站立', prompt)
        self.assertIn('躯干朝向画面左侧的人物', prompt)
        self.assertIn('肩部齐平', prompt)
        self.assertNotIn('\n', external)
        self.assertNotIn('visible_only', prompt)

    def test_missing_or_leaning_posture_is_not_filled_in(self):
        for posture in (None, '', 'leaning', '站立'):
            value = fixture()
            if posture is None:
                del value['people'][1]['posture']
            else:
                value['people'][1]['posture'] = posture
            with self.subTest(posture=posture), self.assertRaisesRegex(ValueError, 'posture'):
                plan_module.assemble(self.encoded(value))

    def test_added_unknown_pose_and_rule_labels_rejected(self):
        for key, new in [('posture', 'visible_only'), ('label', '稳定的人（男子）')]:
            value = fixture(); value['people'][1][key] = new
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan_module.assemble(self.encoded(value))

    def test_unobserved_reference_posture_not_forced_to_sitting(self):
        value = fixture()
        external, _ = plan_module.assemble(self.encoded(value))
        self.assertNotIn('坐', json.loads(external)['prompt'])
        value['people'][0]['posture_detail'] = ''
        with self.assertRaisesRegex(ValueError, 'posture_detail'):
            plan_module.assemble(self.encoded(value))

    def test_repeated_display_names_keep_ids_ownership_and_unique_labels(self):
        value = fixture()
        value['people'][0]['label'] = value['people'][1]['label'] = '人物'
        normalized = plan_module.normalized_plan(self.encoded(value))
        plan = json.loads(normalized)
        self.assertEqual([p['label'] for p in plan['people']], ['人物（1）', '人物（2）'])
        self.assertEqual([p['id'] for p in plan['people']], ['p1', 'p2'])
        self.assertEqual(plan['objects'][0]['owner_id'], 'p2')
        self.assertEqual(plan_module.normalized_plan(normalized), normalized)
        self.assertIn('人物（2）手持的书', plan_module.assemble(normalized)[1])

    def test_display_name_normalization_avoids_existing_name_collisions(self):
        value = fixture()
        value['people'][0]['label'] = value['people'][1]['label'] = '人物'
        extra = dict(value['people'][1], id='p3', label='人物（1）')
        value['people'].append(extra)
        plan = plan_module.validate_plan(self.encoded(value))
        self.assertEqual([p['label'] for p in plan['people']], ['人物（2）', '人物（3）', '人物（1）'])

    def test_abdomen_is_valid_height_but_unknown_camera_side_still_fails(self):
        value = fixture(); value['camera']['height'] = '高度与人物腹部齐平'
        self.assertIn('腹部齐平', plan_module.assemble(self.encoded(value))[1])
        value['camera']['direction'] = '稍微侧向移动，以覆盖互动空间'
        with self.assertRaisesRegex(ValueError, 'camera.direction'):
            plan_module.assemble(self.encoded(value))

    def test_missing_height_generic_visibility_and_unspecified_side_rejected(self):
        for key, new in [('height', ''), ('height', '与坐姿持平'),
                         ('direction', '从侧面观察'), ('visibility', '完整清晰展示所有互动')]:
            value = fixture(); value['camera'][key] = new
            with self.subTest(key=key, new=new), self.assertRaisesRegex(ValueError, 'camera'):
                plan_module.assemble(self.encoded(value))

    def test_literal_waist_requirement_cannot_become_handheld(self):
        value = fixture()
        value['objects'][0]['name'] = '工具包'
        request = '让人物从另一个人腰间的工具包中取出卷尺'
        with self.assertRaisesRegex(ValueError, '腰间归属'):
            plan_module.assemble(self.encoded(value), request)
        value['objects'][0]['mount'] = 'waist'
        self.assertIn('腰间固定的工具包', plan_module.assemble(self.encoded(value), request)[1])

    def test_invalid_owner_and_contact_reference_rejected(self):
        for change in ('owner', 'contact', 'person_id'):
            value = fixture()
            if change == 'owner': value['objects'][0]['owner_id'] = 'missing'
            elif change == 'contact': value['contacts'][0]['source']['id'] = 'missing'
            else: value['people'][1]['id'] = 'p1'
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan_module.assemble(self.encoded(value))

    def test_duplicate_keys_truncation_markers_and_extra_fields_rejected(self):
        for text in ('{"version":1,"version":1}', self.encoded()[:-1],
                     '<think>reason</think>' + self.encoded(),
                     self.encoded(dict(fixture(), prompt='unexpected'))):
            with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                plan_module.assemble(text)

    def test_local_edit_does_not_invent_people_or_camera(self):
        value = dict(version=1, scope='local', goal='将外套改为红色', people=[], objects=[],
                     contacts=[], camera=dict(mode='retain', direction='', height='', framing='', visibility=''),
                     retain='保留外套款式与原构图')
        prompt = plan_module.assemble(self.encoded(value))[1]
        self.assertEqual(prompt, '将外套改为红色。保留外套款式与原构图。')
        value['people'] = [fixture()['people'][1]]
        with self.assertRaisesRegex(ValueError, 'scope'):
            plan_module.assemble(self.encoded(value))


class StructuredRuntimeTests(unittest.TestCase):
    setUp = legacy.InferenceAdapterTests.setUp
    tearDown = legacy.InferenceAdapterTests.tearDown
    config = image_tests.ImageTaskTests.config
    def plan(self, answer=None, finish='stop', **overrides):
        answer = json.dumps(fixture(), ensure_ascii=False) if answer is None else answer
        self.response = {'choices': [{'finish_reason': finish, 'message': {'content': answer}}]}
        values = dict(config_override=self.config(), verified_facts='', user_prompt='让两个人交接一本书',
                      seed=42, image=self.args['first_frame'])
        values.update(overrides)
        return nodes.SXDGemmaSpatialPlan().run(**values)

    def test_one_inference_shared_cache_and_no_inference_during_assembly(self):
        result = self.plan()
        self.assertEqual(self.events.count('generate'), 1)
        first = self.instance
        self.assertEqual(json.loads(result[0])['version'], 1)
        self.assertEqual(result[2], '让两个人交接一本书')
        output = nodes.SXDSpatialPlanAssemble().run(result[0], result[2])
        self.assertEqual(set(json.loads(output[0])), {'prompt'})
        self.assertEqual(self.events.count('generate'), 1)
        self.assertIs(self.instance, first)
        self.assertIn('structured plan', self.call['messages'][0]['content'])

    def test_plan_validation_and_truncation_fail_before_assembly(self):
        value = fixture(); del value['people'][1]['posture']
        for answer, finish in [(json.dumps(value), 'stop'),
                               (json.dumps(fixture()), 'length')]:
            with self.subTest(finish=finish), self.assertRaises(RuntimeError):
                self.plan(answer, finish)
            self.assertIsNone(legacy.adapter._RESIDENT)

    def test_whole_fenced_plan_uses_same_validator_and_one_model_call(self):
        for opening, newline in [('```json', '\n'), ('```JSON', '\r\n'), ('```', '\n')]:
            before = self.events.count('generate')
            result = self.plan(opening + newline + json.dumps(fixture()) + newline + '```')
            self.assertEqual(json.loads(result[0])['version'], 1)
            self.assertFalse(result[0].startswith('```'))
            self.assertEqual(self.events.count('generate') - before, 1)

    def test_shared_fence_handling_preserves_prompt_and_anatomy_tasks(self):
        for task, value in [('prompt', {'prompt': '将外套改为红色。'}),
                            ('anatomy', {'status': 'uncertain', 'issues': []})]:
            text = '```json\n' + json.dumps(value) + '\n```'
            self.assertEqual(json.loads(nodes.validate_answer(text, task)), value)

    def test_fence_does_not_repair_malformed_json_or_missing_semantic_fields(self):
        body = json.dumps(fixture())
        missing = fixture(); del missing['people'][1]['posture']
        for text in ['```json\n' + body, 'Explanation\n```json\n' + body + '\n```',
                     '```json\n' + body + '\n```\n```json\n' + body + '\n```',
                     '```json\n' + body[:-1] + '\n```',
                     '```json\n' + json.dumps(missing) + '\n```']:
            with self.subTest(text=text[:30]), self.assertRaises(RuntimeError):
                self.plan(text)
            self.assertIsNone(legacy.adapter._RESIDENT)

    def test_exactly_one_reference_image_required(self):
        for changes in (dict(image=None), dict(image2=self.args['first_frame'])):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.plan(**changes)
        self.assertEqual(self.events.count('generate'), 0)

    def test_verified_facts_do_not_replace_original_request_output(self):
        result = self.plan(verified_facts='上身直立，完整站姿或坐姿无法确认')
        content = self.call['messages'][1]['content'][0]['text']
        self.assertIn('Verified reference facts', content)
        self.assertEqual(result[2], '让两个人交接一本书')


class PublishedWorkflowTests(unittest.TestCase):
    def setUp(self):
        path = Path(plan_module.__file__).parent / 'workflows/qwen/image2image_qwen2.1-gemma-v9-live-structured.json'
        self.text = path.read_text(encoding='utf-8')
        self.workflow = json.loads(self.text)
        self.nodes = {n['id']: n for n in self.workflow['nodes']}
        self.links = {link[0]: link for link in self.workflow['links']}

    def test_all_links_are_bidirectional(self):
        self.assertEqual(len(self.nodes), len(self.workflow['nodes']))
        self.assertEqual(len(self.links), len(self.workflow['links']))
        for link in self.links.values():
            ident, source, output, target, input_slot, _ = link
            self.assertIn(ident, self.nodes[source]['outputs'][output]['links'])
            self.assertEqual(self.nodes[target]['inputs'][input_slot]['link'], ident)
        for node in self.nodes.values():
            for index, item in enumerate(node.get('inputs', [])):
                if item.get('link') is not None:
                    self.assertEqual(self.links[item['link']][3:5], [node['id'], index])
            for index, item in enumerate(node.get('outputs', [])):
                for ident in item.get('links') or []:
                    self.assertEqual(self.links[ident][1:3], [node['id'], index])

    def test_only_assembled_json_reaches_original_extractor(self):
        self.assertEqual(self.nodes[1151]['type'], 'SXDGemmaSpatialPlan')
        self.assertEqual(self.links[115][1:5], [1151, 0, 1361, 0])
        self.assertEqual(self.links[116][1:5], [1361, 0, 1360, 0])
        self.assertEqual(self.links[42][1:5], [1360, 0, 1213, 0])
        self.assertEqual(self.links[34][1:5], [1213, 0, 1204, 0])
        self.assertEqual(self.links[117][1:5], [1151, 2, 1360, 1])

    def test_live_displays_keep_raw_slot_one(self):
        self.assertEqual(self.links[113][1:5], [1151, 1, 1358, 0])
        self.assertEqual(self.links[114][1:5], [1152, 1, 1359, 0])
        self.assertEqual(self.nodes[1152]['type'], 'SXDGemmaImageTask')

    def test_user_inputs_are_blank_and_model_paths_relative(self):
        self.assertEqual(self.nodes[1109]['widgets_values'][0], '')
        self.assertEqual(self.nodes[1109]['widgets_values_named']['string'], '')
        self.assertEqual(self.nodes[1200]['widgets_values'][0], '')
        self.assertEqual(self.nodes[1201]['widgets_values'][0], '')
        self.assertNotIn('r2.spellpic.com', self.text)
        config = json.loads(self.nodes[1150]['widgets_values'][0])
        self.assertFalse(Path(config['model_path']).is_absolute())
        self.assertFalse(Path(config['mmproj_path']).is_absolute())

    def test_saved_widget_aliases_match_visible_defaults(self):
        encoder = self.nodes[14]
        self.assertEqual(encoder['widgets_values_named']['prompt'], encoder['widgets_values'][0])
        self.assertEqual(encoder['widgets_values_named']['negative_prompt'], encoder['widgets_values'][1])
        self.assertEqual(self.nodes[1151]['inputs'][1]['name'], 'verified_facts')
        self.assertEqual(self.nodes[1151]['widgets_values'], [False])


if __name__ == '__main__':
    unittest.main()

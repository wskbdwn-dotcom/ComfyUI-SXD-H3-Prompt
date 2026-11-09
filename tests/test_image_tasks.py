"""Native image nodes exercise the same runtime/cache as existing H3 tests."""
import base64
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
from PIL import Image
import test_v2 as legacy

nodes = sys.modules[legacy.spec.name + '.image_nodes']
trace_module = sys.modules[legacy.spec.name + '.image_trace']
runtime = legacy.adapter
ANSWER = '{"prompt":"保持人物身份，改穿蓝色外套。"}'


class ImageTaskTests(unittest.TestCase):
    setUp = legacy.InferenceAdapterTests.setUp
    tearDown = legacy.InferenceAdapterTests.tearDown

    def config(self, **values):
        config = dict(model_path=str(Path(self.tmp.name) / 'LLM/gemma-4-26b.gguf'),
                      mmproj_path=str(Path(self.tmp.name) / 'LLM/mmproj.gguf'))
        config.update(values)
        return json.dumps(config)

    def run_task(self, answer=ANSWER, finish='stop', **overrides):
        self.response = {'choices': [{'finish_reason': finish, 'message': {'content': answer}}]}
        args = dict(config_override=self.config(), system_prompt_override='SYSTEM RULES',
                    user_prompt='换成蓝色外套', seed=99, image=self.args['first_frame'])
        args.update(overrides)
        return nodes.SXDGemmaImageTask().run(**args)

    def test_live_success_and_error_keep_raw_output(self):
        events = []
        server = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(
            client_id='browser', send_sync=lambda event, data, client: events.append((event, data, client)))))
        with mock.patch.dict(sys.modules, {'server': server}):
            result = self.run_task('<think>检查衣服。</think>' + ANSWER, unique_id='1151')
            self.assertEqual(events[0][1]['status'], 'starting')
            self.assertEqual(events[-1][1]['status'], 'done')
            self.assertTrue(all(e[1]['node'] == '1151' and e[2] == 'browser' for e in events))
            visible = ''.join(e[1]['delta'] for e in events)
            self.assertEqual(visible, result[1])
            self.assertEqual(visible.count('检查衣服。'), 1)
            self.assertEqual(result[0], ANSWER)
            events.clear()
            with self.assertRaises(RuntimeError):
                self.run_task('<think>部分思考', finish='length', unique_id='1152')
            self.assertEqual(events[-1][1]['status'], 'error_or_cancelled')
            self.assertIn('部分思考', ''.join(e[1]['delta'] for e in events))

    def test_parsed_only_stream_is_visible_before_completion(self):
        trace = trace_module.Trace({}, validate=lambda value: value, node_id='1151')
        events = []
        trace.preview.sender = lambda event, data, client: events.append(data)
        trace.send('starting')
        trace.record({'choices': [{'text': ''}]}, True)
        def chunks():
            trace.preview.last_send = 0
            yield {'choices': [{'delta': {'reasoning_content': '先确认动作。'}}]}
            self.assertIn('先确认动作。', ''.join(e['delta'] for e in events))
            yield {'choices': [{'delta': {'content': ANSWER}, 'finish_reason': 'stop'}]}
        result = trace.collect(chunks())
        trace.send('done')
        self.assertEqual(result['choices'][0]['message']['content'], ANSWER)
        self.assertEqual(trace.raw_output, '<think>先确认动作。</think>' + ANSWER)
        self.assertEqual(''.join(e['delta'] for e in events), trace.raw_output)

    def test_preview_disconnect_does_not_break_inference(self):
        trace = trace_module.Trace({})
        trace.preview.sender = mock.Mock(side_effect=RuntimeError('disconnected'))
        with self.assertLogs(trace_module.LiveTrace.__module__, level='WARNING'):
            trace.send('starting')
        self.assertIsNone(trace.preview.sender)
        result = trace.collect(iter([{'choices': [{'delta': {'content': ANSWER}, 'finish_reason': 'stop'}]}]))
        self.assertEqual(result['choices'][0]['message']['content'], ANSWER)

    def test_native_prompt_and_anatomy_share_h3_cache(self):
        legacy.nodes.H3AuthorGemma4ExpandV2().expand(**self.args, enable_thinking=True, n_ctx=32768)
        original = self.instance
        result = self.run_task('<think>检查衣服。</think>' + ANSWER)
        self.assertEqual(result[0], ANSWER)
        self.assertIn('<think>', result[1])
        self.assertIs(self.instance, original)
        self.assertEqual(self.run_task('{"issues":[],"status":"uncertain"}', task='anatomy')[0],
                         '{"status":"uncertain","issues":[]}')
        self.assertIs(self.instance, original)
        self.assertEqual(self.penalty_last_n, 4096)
        self.assertEqual(self.call['temperature'], 1.0)
        self.assertEqual(self.call['messages'][0]['content'], 'SYSTEM RULES')
        self.assertEqual(sum(isinstance(e, tuple) and e[0] == 'load_llama' for e in self.events), 1)
        self.assertEqual(self.instance._ctx.tokens, [])

    def test_relative_models_supported(self):
        self.run_task(config_override=self.config(model_path='gemma-4-26b.gguf', mmproj_path='mmproj.gguf'))
        self.assertIsNotNone(runtime._RESIDENT)

    def test_changed_sampling_reuses_but_changed_context_reloads(self):
        self.run_task(); first = self.instance
        self.run_task(config_override=self.config(temperature=.7, penalty_last_n=2048))
        self.assertIs(self.instance, first)
        self.assertEqual(self.penalty_last_n, 2048)
        self.run_task(config_override=self.config(n_ctx=16384))
        self.assertIsNot(self.instance, first)

    def test_release_switch(self):
        self.run_task(release_after_run=True)
        self.assertIsNone(runtime._RESIDENT)
        self.assertIn('close_llama', self.events)

    def test_invalid_json_and_truncation_release_without_pass(self):
        for answer, finish in [(ANSWER, 'length'), ('not json', 'stop'),
                               ('{"status":"pass","issues":[]}', 'stop'), (ANSWER, None)]:
            with self.subTest(answer=answer, finish=finish), self.assertRaises(RuntimeError):
                self.run_task(answer, finish)
            self.assertIsNone(runtime._RESIDENT)
            self.assertNotIn('create_completion', vars(self.instance))

    def test_cancellation_releases_and_restores(self):
        self.cancel_during_generation = True
        with self.assertRaises(self.mm.InterruptProcessingException): self.run_task()
        self.assertIsNone(runtime._RESIDENT)
        self.assertNotIn('create_completion', vars(self.instance))

    def test_repetition_stops_and_releases(self):
        repeated = 'Check the body ownership and connection before deciding this same step.\n' * 20
        with self.assertRaisesRegex(RuntimeError, 'THINKING_LOOP'):
            self.run_task('<think>' + repeated + '</think>' + ANSWER)
        self.assertIsNone(runtime._RESIDENT)

    def test_invalid_config_or_paths_never_load(self):
        for changes in ({'script': 'qwen3vl_run.py'}, {'enable_thinking': 'true'},
                        {'top_k': 1.5}, {'temperature': float('nan')},
                        {'max_tokens':32768}, {'model_path':'/missing/model.gguf'}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, FileNotFoundError)):
                self.run_task(config_override=self.config(**changes))
        self.assertNotIn('generate', self.events)

    def test_image_count_and_anatomy_single_image_gate(self):
        batch = legacy.Tensor(np.zeros((4, 20, 40, 3), np.float32))
        self.run_task(image=batch)
        urls = [x for x in self.call['messages'][1]['content'] if x['type'] == 'image_url']
        self.assertEqual(len(urls), 4)
        for image in (None, batch, legacy.Tensor(np.zeros((5,20,40,3),np.float32))):
            with self.assertRaises(ValueError): self.run_task(task='anatomy', image=image)

    def test_image_preprocessing_preserves_aspect_and_caps_long_edge(self):
        image = legacy.Tensor(np.ones((1, 900, 1800, 3), np.float32))
        self.run_task(image=image)
        item = self.call['messages'][1]['content'][-1]['image_url']['url']
        decoded = Image.open(io.BytesIO(base64.b64decode(item.split(',', 1)[1])))
        self.assertEqual(decoded.size, (1024,512))

    def test_four_separate_inputs_preserve_order_and_total_limit(self):
        images = [legacy.Tensor(np.full((1,20,40,3), x, np.float32)) for x in (0,.25,.5,1)]
        self.run_task(image=images[0], image2=images[1], image3=images[2], image4=images[3])
        content = self.call['messages'][1]['content']
        labels = [c['text'] for c in content[1:] if c['type']=='text']
        self.assertEqual(labels, ['Image-1','Image-2','Image-3','Image-4'])
        means = []
        for c in content:
            if c['type']=='image_url':
                picture = Image.open(io.BytesIO(base64.b64decode(c['image_url']['url'].split(',',1)[1])))
                means.append(np.array(picture).mean())
        self.assertEqual(means, sorted(means)); self.assertEqual(len(set(means)),4)
        with self.assertRaises(ValueError):
            self.run_task(image=legacy.Tensor(np.zeros((4,20,40,3),np.float32)),image2=images[1])

    def test_strict_anatomy_schema(self):
        issue = dict(type='extra_arm', location='左侧白衣人物肩部', evidence='同一肩部连出两条手臂', fix='移除额外手臂，其他部分不变')
        valid = json.dumps(dict(issues=[issue], status='repair'), ensure_ascii=False)
        result = self.run_task(valid, task='anatomy')[0]
        self.assertEqual(list(json.loads(result)), ['status','issues'])
        self.assertEqual(list(json.loads(result)['issues'][0]), ['type','location','evidence','fix'])
        for value in (dict(status='repair', issues=[]), dict(status='uncertain', issues=[issue]),
                      dict(status='pass', issues=[], score=1),
                      dict(status='repair', issues=[dict(issue, type='extra_hand')])):
            with self.assertRaises(RuntimeError): self.run_task(json.dumps(value), task='anatomy')
            self.assertIsNone(runtime._RESIDENT)

    def test_duplicate_fields_rejected(self):
        with self.assertRaises(RuntimeError): self.run_task('{"prompt":"a","prompt":"b"}')
        self.assertIsNone(runtime._RESIDENT)

    def test_file_change_invalidates_node_cache(self):
        args = dict(config_override=self.config())
        before = nodes.SXDGemmaImageTask.IS_CHANGED(**args)
        (Path(self.tmp.name)/'LLM/gemma-4-26b.gguf').write_bytes(b'updated model identity')
        self.assertNotEqual(before, nodes.SXDGemmaImageTask.IS_CHANGED(**args))


class ImageTraceTests(unittest.TestCase):
    def collect(self, text, step=5, finish='stop', separate=False):
        trace = trace_module.Trace({}, cancel=lambda:None)
        def native():
            for i in range(0, len(text), step):
                yield {'choices':[{'text':text[i:i+step], 'finish_reason':None}]}
            yield {'choices':[{'text':'','finish_reason':finish}]}
        def chat():
            for chunk in trace.observe(native()):
                choice=chunk['choices'][0]
                delta={'reasoning_content':choice['text']} if separate else {'content':''}
                yield {'choices':[{'delta':delta,'finish_reason':choice['finish_reason']}]}
            if separate:
                yield {'choices':[{'delta':{'content':ANSWER},'finish_reason':'stop'}]}
        return trace.collect(chat())['choices'][0]['message']['content']

    def test_every_marker_split_and_native_final_chunk_preserved(self):
        for step in range(1,25):
            for opening,closing in [('<think>','</think>'),('<|channel>thought','<channel|>')]:
                with self.subTest(step=step,opening=opening):
                    self.assertEqual(self.collect(opening+'检查人物'+closing+ANSWER,step),ANSWER)

    def test_no_newline_and_separate_reasoning_loops(self):
        block='Repeated long reasoning about identities and relationships without newlines. '
        for text,separate in [('<think>'+block*20+'</think>'+ANSWER,False),(block*20,True)]:
            with self.assertRaisesRegex(RuntimeError,'THINKING_LOOP'): self.collect(text,separate=separate)

    def test_final_repetition_and_literal_markers_allowed(self):
        answer=json.dumps({'prompt':'Write <think> on a sign. '+'Repeated long final caption intended for the image. '*30})
        self.assertEqual(self.collect(answer),answer)

    def test_plain_thought_closed_by_channel_keeps_the_json(self):
        body = ('{"人数": 2, "alias": "", "接触": "阴茎进入阴道", "支撑": "跪", '
                '"轴线": "男方在女方身后", "body_parts": {"女方": {"头": "", "胸": "", '
                '"腰": "", "胯": "", "臀": "", "大腿": "", "膝盖": "", "脚": ""}, '
                '"男方": {"头": "", "胸": "", "腰": "", "胯": "", "臀": "", "大腿": "", '
                '"膝盖": "", "脚": ""}}}')
        thought = 'The user said kneeling from behind. Valid.'
        self.assertEqual(self.collect(thought + '<channel|>' + body), body)
        self.assertEqual(self.collect('earlier <channel|> still thinking <channel|>' + body), body)

    def test_separate_reasoning_is_not_final_text(self):
        self.assertEqual(self.collect('Brief reasoning.',separate=True),ANSWER)

    def test_incomplete_or_empty_responses_rejected(self):
        for text,finish in [(ANSWER,'length'),(ANSWER,None),(ANSWER,'tool_calls'),('<think>unfinished','stop'),('<think>done</think>','stop')]:
            with self.subTest(text=text,finish=finish),self.assertRaises(RuntimeError): self.collect(text,finish=finish)


if __name__ == '__main__': unittest.main()

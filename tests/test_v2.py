"""Offline tests: real image preprocessing, simulated Comfy/llama inference.

No models are loaded and no GPU or ComfyUI server is exercised.
Run with a Python environment containing numpy and Pillow.
"""
import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT
NAME = 'image2video-minimax_h3_v1'

folder_paths = types.ModuleType('folder_paths')
folder_paths.models_dir = ''
folder_paths.get_folder_paths = lambda _: []
sys.modules['folder_paths'] = folder_paths
spec = importlib.util.spec_from_file_location('h3_author_gemma4_test', PLUGIN / '__init__.py', submodule_search_locations=[str(PLUGIN)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
nodes = sys.modules[spec.name + '.nodes']
adapter = nodes.runtime


class Tensor:
    def __init__(self, values): self.values = values
    @property
    def shape(self): return self.values.shape
    def __getitem__(self, key): return Tensor(self.values[key])
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.values


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class InferenceAdapterTests(unittest.TestCase):
    def test_thinking_budget_schema_and_forwarding(self):
        schema = nodes.H3AuthorGemma4ExpandV2.INPUT_TYPES()['required']['max_tokens'][1]
        self.assertEqual(schema['default'], 4096)
        self.assertEqual(schema['max'], 1048576)
        context_schema = nodes.H3AuthorGemma4ExpandV2.INPUT_TYPES()['required']['n_ctx'][1]
        self.assertEqual(context_schema['max'], 1048576)
        self.assertEqual(context_schema['default'], 16384)
        adapter.H3AuthorGemma4Expand().expand(**self.args, enable_thinking=True,
                                            max_tokens=65536, n_ctx=262144)
        self.assertEqual(self.call['max_tokens'], 65536)
        loads = [event[1] for event in self.events if isinstance(event, tuple) and event[0] == 'load_llama']
        self.assertEqual(loads[-1]['n_ctx'], 262144)
        self.assertIn(('thinking', True), self.events)
        self.assertEqual(self.instance._ctx.tokens, [])

    def test_truncation_reports_budget_without_publishing_partial_content(self):
        self.response = {'choices': [{'finish_reason': 'length', 'message': {
            'content': 'PRIVATE PARTIAL', 'reasoning_content': 'PRIVATE REASONING'}}],
            'usage': {'prompt_tokens': 6000, 'completion_tokens': 8192, 'total_tokens': 14192}}
        with self.assertRaises(RuntimeError) as caught:
            adapter.H3AuthorGemma4Expand().expand(**self.args, enable_thinking=True,
                                                max_tokens=8192, n_ctx=32768)
        message = str(caught.exception)
        for expected in ('max_tokens=8192', 'n_ctx=32768', 'enable_thinking=True',
                         'prompt_tokens=6000', 'completion_tokens=8192', 'total_tokens=14192'):
            self.assertIn(expected, message)
        self.assertNotIn('PRIVATE', message)
        self.assertNotIn('关闭 enable_thinking', message)
        self.assertIsNone(adapter._RESIDENT)

    def test_truncation_missing_usage_is_not_reported_as_zero(self):
        for usage in (None, {}, 'invalid', {'prompt_tokens': 'PRIVATE'}):
            response = {'choices': [{'finish_reason': 'length'}], 'usage': usage}
            with self.assertRaises(RuntimeError) as caught:
                adapter._final_content(response)
            self.assertIn('prompt_tokens=未提供', str(caught.exception))
            self.assertNotIn('PRIVATE', str(caught.exception))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        folder_paths.models_dir = self.tmp.name
        modeldir = Path(self.tmp.name) / 'LLM'
        modeldir.mkdir()
        for filename in ('gemma-4-26b.gguf', 'mmproj.gguf'): (modeldir / filename).write_bytes(b'test placeholder')
        self.events = []
        self.response = {'choices':[{'finish_reason':'stop','message':{'content':'integrated_multimodal_description: [Shot 1] A person waves. overall_soundscape: Wind. non_diegetic_music: N/A'}}]}
        self.mm = types.ModuleType('comfy.model_management')
        self.mm.processing_interrupted = lambda: False
        self.mm.InterruptProcessingException = type('Cancelled', (BaseException,), {})
        self.mm.unload_all_models = lambda: self.events.append('unload_comfy')
        self.mm.soft_empty_cache = lambda: None
        comfy = types.ModuleType('comfy')
        comfy.model_management = self.mm
        self.modules = mock.patch.dict(sys.modules, {'comfy':comfy,'comfy.model_management':self.mm})
        self.modules.start()
        self.args = dict(prompt='挥手', first_frame=Tensor(np.zeros((1,20,40,3),np.float32)), frame_count=124,
                         seed=2**40+7, model='gemma-4-26b.gguf', mmproj='mmproj.gguf')
        parent = self
        class BaseHandler:
            def __init__(self, mmproj_path, verbose=False): parent.events.append(('projector',mmproj_path))
        class Handler(BaseHandler):
            def __init__(self, enable_thinking=True, **kwargs):
                parent.events.append(('thinking',enable_thinking))
                super().__init__(**kwargs)
            def close(self): parent.events.append('close_handler')
        class NativeContext:
            def __init__(self): self.tokens = []
            def memory_clear(self, data):
                parent.assertIs(data, True)
                self.tokens.clear()
                parent.events.append('native_clear')
        class Checkpoints:
            def __init__(self): self.saved = []
            def clear(self):
                self.saved.clear()
                parent.events.append('checkpoint_clear')
        class Llama:
            def __init__(self, **kwargs):
                parent.events.append(('load_llama',kwargs))
                parent.instance = self
                self.state = []
                self._ctx = NativeContext()
                self._hybrid_cache_mgr = Checkpoints()
            def reset(self):
                self.state.clear()
                parent.events.append('reset')
            # Deliberately reject unknown keywords: the real chat entry point
            # does NOT accept stopping_criteria (unlike create_completion).
            def create_chat_completion(self, messages, seed, max_tokens,
                                       temperature, top_p):
                if self._ctx.tokens:
                    raise RuntimeError('Fatal Decode Error at Pos 0: stale native state')
                parent.assertEqual(self._hybrid_cache_mgr.saved, [])
                self._ctx.tokens.append('evaluated text/image tokens')
                self._hybrid_cache_mgr.saved.append('native checkpoint')
                parent.assertEqual(self.state, [])
                self.state.append(messages)
                parent.call = dict(messages=messages, seed=seed, max_tokens=max_tokens,
                                   temperature=temperature, top_p=top_p)
                return self.create_completion(prompt='rendered image + prompt',
                    stopping_criteria=[lambda *args: False])
            def create_completion(self, prompt, stopping_criteria=None):
                parent.events.append('generate')
                parent.criteria = stopping_criteria
                if getattr(parent, 'cancel_during_generation', False):
                    parent.mm.processing_interrupted = lambda: True
                for criterion in stopping_criteria or []:
                    criterion(None, None)
                if isinstance(parent.response, Exception): raise parent.response
                return parent.response
            def close(self): parent.events.append('close_llama')
        self.Handler, self.Llama = Handler, Llama
        self.backend = mock.patch.object(adapter, '_backend', return_value=(Llama,Handler,list))
        self.backend.start()

    def tearDown(self):
        with adapter._LOCK:
            adapter._release_resident()
        self.addCleanup(self.assertNotIn, "unload_comfy", self.events)
        self.backend.stop()
        self.modules.stop()
        self.tmp.cleanup()

    def test_repeated_expansion_preserves_comfy_models(self):
        resident = [object()]
        original = resident[0]
        def unload():
            resident.clear()
            self.events.append('unload_comfy')
        self.mm.unload_all_models = unload
        for _ in range(2):
            adapter.H3AuthorGemma4Expand().expand(**self.args)
            self.assertEqual(resident, [original])
        self.assertEqual(self.events.count('close_llama'), 0)
        self.assertEqual(self.events.count('close_handler'), 0)

    def test_changed_prompt_image_seed_and_output_options_reuse_model(self):
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        instance = self.instance
        args = dict(self.args, prompt='向前走', seed=8,
                    first_frame=Tensor(np.ones((1,20,40,3),np.float32)))
        adapter.H3AuthorGemma4Expand().expand(**args, max_tokens=512, output_language='English')
        self.assertIs(instance, self.instance)
        self.assertEqual(self.instance.state, [])
        self.assertIn('向前走', self.call['messages'][1]['content'][0]['text'])
        self.assertEqual(self.call['seed'], 8)
        self.assertEqual(self.events.count('reset'), 4)
        self.assertNotIn('create_completion', vars(instance))

    def test_loading_configuration_changes_replace_single_pair(self):
        args = dict(self.args)
        for field, value in [(None,None),('n_ctx',8192),('n_gpu_layers',20),('enable_thinking',True),
                             ('model','other.gguf'),('mmproj','other-projector.gguf')]:
            if field in ('model','mmproj'):
                (Path(self.tmp.name)/'LLM'/value).write_bytes(b'another model')
            if field: args[field] = value
            adapter.H3AuthorGemma4Expand().expand(**args)
        self.assertEqual(sum(isinstance(e,tuple) and e[0]=='load_llama' for e in self.events), 6)
        self.assertEqual(self.events.count('close_llama'), 5)
        self.assertEqual(self.events.count('close_handler'), 5)
        self.assertIs(adapter._RESIDENT[1], self.instance)

    def test_changed_model_file_reloads(self):
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        old = self.instance
        (Path(self.tmp.name)/'LLM'/self.args['model']).write_bytes(b'new model contents')
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIsNot(old, self.instance)
        self.assertEqual(self.events.count('close_llama'),1)

    def test_manual_release_and_next_run_reload(self):
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        old = self.instance
        release = adapter.H3AuthorGemma4Release()
        release.release(False)
        self.assertIs(adapter._RESIDENT[1],old)
        release.release(True)
        release.release(True)
        self.assertIsNone(adapter._RESIDENT)
        self.assertEqual(self.events.count('close_llama'),1)
        self.assertEqual(self.events.count('close_handler'),1)
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIsNot(old,self.instance)
        self.assertTrue(np.isnan(release.IS_CHANGED()))

    def test_error_invalidates_cache_and_recovers_on_next_run(self):
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        old = self.instance
        good_response = self.response
        self.response = RuntimeError('failed native call')
        with self.assertRaisesRegex(RuntimeError, 'failed native call'):
            adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIsNone(adapter._RESIDENT)
        self.response = good_response
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIsNot(old,self.instance)

    def test_pending_cancel_preserves_healthy_cache(self):
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        old = self.instance
        self.mm.processing_interrupted = lambda: True
        with self.assertRaises(self.mm.InterruptProcessingException):
            adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIs(adapter._RESIDENT[1],old)
        self.assertNotIn('close_llama',self.events)

    def test_reset_failure_invalidates_cache(self):
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        with mock.patch.object(self.instance, 'reset', side_effect=RuntimeError('reset failed')):
            with self.assertRaisesRegex(RuntimeError, 'reset failed'):
                adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIsNone(adapter._RESIDENT)
        self.assertEqual(self.events[-2:], ['close_handler','close_llama'])

    def test_release_switch_is_optional_and_defaults_to_keep(self):
        schema = adapter.H3AuthorGemma4Expand.INPUT_TYPES()
        self.assertNotIn('release_after_expansion', schema['required'])
        self.assertIs(schema['optional']['release_after_expansion'][1]['default'], False)
        # Old API calls omit the new field and keep a healthy model resident.
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIsNotNone(adapter._RESIDENT)

    def test_release_switch_closes_after_success_and_next_call_loads(self):
        node = adapter.H3AuthorGemma4Expand()
        first = node.expand(**self.args, release_after_expansion=True)
        self.assertIsNone(adapter._RESIDENT)
        self.assertEqual(self.events[-2:], ['close_handler', 'close_llama'])
        old = self.instance
        second = node.expand(**self.args, release_after_expansion=False)
        self.assertEqual(first, second)
        self.assertIsNot(old, self.instance)
        self.assertIs(adapter._RESIDENT[1], self.instance)
        self.assertEqual(self.events.count('close_llama'), 1)

    def test_release_switch_reuses_then_releases_existing_pair(self):
        node = adapter.H3AuthorGemma4Expand()
        node.expand(**self.args)
        original = self.instance
        node.expand(**self.args, release_after_expansion=True)
        self.assertIs(original, self.instance)
        self.assertIsNone(adapter._RESIDENT)
        self.assertEqual(sum(isinstance(e,tuple) and e[0]=='load_llama' for e in self.events),1)
        self.assertEqual(self.events.count('close_llama'),1)
        self.assertEqual(self.events.count('close_handler'),1)

    def test_legacy_reset_reproduces_second_run_failure(self):
        # The previous implementation cleared Python state only, as in 0.3.35.
        with mock.patch.object(adapter, '_reset_inference_state', side_effect=lambda llm: llm.reset()):
            adapter.H3AuthorGemma4Expand().expand(**self.args)
            self.assertTrue(self.instance._ctx.tokens)
            with self.assertRaisesRegex(RuntimeError, 'Fatal Decode Error at Pos 0'):
                adapter.H3AuthorGemma4Expand().expand(**dict(self.args,seed=8))
        self.assertIsNone(adapter._RESIDENT)

    def test_native_state_and_checkpoints_clear_without_reloading(self):
        node = adapter.H3AuthorGemma4Expand()
        for seed in (7,8,9):
            node.expand(**dict(self.args, seed=seed))
            self.assertEqual(self.instance._ctx.tokens, [])
            self.assertEqual(self.instance._hybrid_cache_mgr.saved, [])
        self.assertEqual(sum(isinstance(e,tuple) and e[0]=='load_llama' for e in self.events),1)
        self.assertEqual(self.events.count('close_llama'),0)
        self.assertEqual(self.events.count('native_clear'),6)
        self.assertEqual(self.events.count('checkpoint_clear'),6)

    def test_native_clear_failure_invalidates_cache(self):
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        with mock.patch.object(self.instance._ctx, 'memory_clear', side_effect=RuntimeError('native clear failed')):
            with self.assertRaisesRegex(RuntimeError,'native clear failed'):
                adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIsNone(adapter._RESIDENT)
        self.assertEqual(self.events[-2:], ['close_handler','close_llama'])

    def test_native_clear_without_hybrid_cache(self):
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.instance._hybrid_cache_mgr = None
        self.instance._ctx.tokens.append('previous native state')
        adapter._reset_inference_state(self.instance)
        self.assertEqual(self.instance._ctx.tokens, [])

    def test_think_toggle_both_directions_and_cleanup(self):
        for enabled in (False, True):
            self.events.clear()
            result = adapter.H3AuthorGemma4Expand().expand(**self.args, enable_thinking=enabled)
            self.assertIn(('thinking',enabled),self.events)
            self.assertEqual(self.events.count('generate'),1)  # No second review model call.
            self.assertEqual(self.call['seed'],7)
            self.assertEqual(self.call['temperature'],0.2)
            self.assertEqual(self.call['top_p'],0.9)
            self.assertIn('duration=5.17s',self.call['messages'][0]['content'])
            self.assertIn('integrated_multimodal_description:\n',result[0])
            self.assertIs(adapter._RESIDENT[1], self.instance)
            self.assertEqual(self.events[-3:], ['reset','native_clear','checkpoint_clear'])

    def test_inference_error_propagates_and_releases(self):
        self.response = RuntimeError('simulated inference failure')
        with self.assertRaisesRegex(RuntimeError,'simulated inference failure'):
            adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertEqual(self.events[-2:],['close_handler','close_llama'])

    def test_load_error_releases_projector(self):
        def broken(**kwargs): raise RuntimeError('simulated model load failure')
        with mock.patch.object(adapter,'_backend',return_value=(broken,self.Handler,list)):
            with self.assertRaisesRegex(RuntimeError,'model load failure'): adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertEqual(self.events[-1:],['close_handler'])

    def test_cancel_does_not_load_model(self):
        self.mm.processing_interrupted = lambda: True
        with self.assertRaises(self.mm.InterruptProcessingException): adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertEqual(self.events,[])

    def test_chat_signature_regression_and_callback_routing(self):
        with self.assertRaisesRegex(TypeError, 'stopping_criteria'):
            self.Llama().create_chat_completion(stopping_criteria=[])
        adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertEqual(len(self.criteria), 2)  # Our interrupt + handler's existing stop.
        self.assertIs(self.instance.create_completion.__func__, self.Llama.create_completion)
        self.assertNotIn('stopping_criteria', self.call)

    def test_cancel_during_generation_restores_method_and_cleans_up(self):
        self.cancel_during_generation = True
        with self.assertRaises(self.mm.InterruptProcessingException):
            adapter.H3AuthorGemma4Expand().expand(**self.args)
        self.assertIs(self.instance.create_completion.__func__, self.Llama.create_completion)
        self.assertEqual(self.events[-2:], ['close_handler','close_llama'])

    def test_missing_model_and_old_projector_signature(self):
        with self.assertRaises(FileNotFoundError): adapter._model_path('missing.gguf')
        with self.assertRaises(ValueError): adapter._model_path('../escape.gguf')
        class Legacy:
            def __init__(self,clip_model_path,verbose=False): pass
        self.assertEqual(adapter._projector_keyword(Legacy),'clip_model_path')

    def test_backend_requires_explicit_think_support(self):
        self.backend.stop()
        try:
            llama = types.ModuleType('llama_cpp')
            llama.__path__ = []
            llama.Llama, llama.StoppingCriteriaList = self.Llama,list
            module = types.ModuleType('llama_cpp.llama_multimodal')
            module.Gemma4ChatHandler = self.Handler
            with mock.patch.dict(sys.modules,{'llama_cpp':llama,'llama_cpp.llama_multimodal':module}):
                self.assertIs(adapter._backend()[1],self.Handler)
                class Unsupported:
                    def __init__(self, **kwargs): pass
                module.Gemma4ChatHandler = Unsupported
                with self.assertRaisesRegex(RuntimeError,'enable_thinking'): adapter._backend()
        finally:
            self.backend.start()



    def test_v2_release_switch_forwarding_in_both_modes(self):
        n = nodes.H3AuthorGemma4ExpandV2()
        for mode in (nodes.KEYFRAMES, nodes.REFERENCES):
            result = n.expand(**self.args, mode=mode, release_after_expansion=True)
            self.assertEqual(len(result),3)
            self.assertIsNone(adapter._RESIDENT)
        self.assertEqual(self.events.count('close_llama'),2)
        self.assertEqual(self.events.count('close_handler'),2)

    def test_v2_routes_both_rules_and_preserves_output_zero(self):
        n = nodes.H3AuthorGemma4ExpandV2()
        result = n.expand(**self.args, last_frame=self.args['first_frame'],
                          keyframe_rules='KEY {{task}} {{labels}}', reference_rules='REF')
        self.assertEqual(result[1], 'KEY FL2VA picture 1, picture 2')
        self.assertIn('picture 2', result[0])  # Author FL2VA postprocessing.
        self.assertEqual(json.loads(result[2])['task'], 'FL2VA')
        result = n.expand(**self.args, mode=nodes.REFERENCES,
                          reference_2=self.args['first_frame'],
                          keyframe_rules='KEY', reference_rules='REF {{labels}}')
        self.assertEqual(result[1], 'REF <picture 1>, <picture 2>')
        self.assertNotIn('参考图像与目标视频对齐关系', result[0])
        self.assertEqual(len(self.call['messages'][1]['content']), 5)

    def test_custom_raw_output_and_preload_validation(self):
        self.response = {'choices':[{'finish_reason':'stop','message':{'content':'free text'}}]}
        result = nodes.H3AuthorGemma4ExpandV2().expand(**self.args,
            keyframe_rules='Output free text.', author_postprocess=False)
        self.assertEqual(result[0], 'free text')
        self.events.clear()
        with self.assertRaises(ValueError):
            nodes.H3AuthorGemma4ExpandV2().expand(**self.args, keyframe_rules='{{bad}}')
        self.assertEqual(self.events, [])


class ModeAndRuleTests(unittest.TestCase):
    def image(self, n=1, value=0):
        return Tensor(np.full((n, 20, 40, 3), value, np.float32))

    def prepare(self, **kwargs):
        args = dict(prompt='走向门口', mode=nodes.KEYFRAMES,
                    keyframe_rules=nodes.DEFAULTS['keyframe_rules'],
                    reference_rules=nodes.DEFAULTS['reference_rules'],
                    frame_count=124, fps=24, language='中文')
        args.update(kwargs)
        return nodes.prepare(**args)

    def test_default_rules_exactly_match_author_all_modes(self):
        count = 0
        for language in ('中文', 'English'):
            for prompt in ('走向门口', '唱歌', '变成金色', '唱歌并替换人物'):
                for frame_count in (81, 124, 240):
                    for mode, first, last in [
                        (nodes.KEYFRAMES, None, None),
                        (nodes.KEYFRAMES, self.image(), None),
                        (nodes.KEYFRAMES, None, self.image()),
                        (nodes.KEYFRAMES, self.image(), self.image()),
                        (nodes.REFERENCES, self.image(), None),
                    ]:
                        task, messages, mapping = self.prepare(mode=mode, first_frame=first,
                            last_frame=last, language=language, prompt=prompt, frame_count=frame_count)
                        m = json.loads(mapping)
                        context = {'main_mode': 'all_reference' if mode == nodes.REFERENCES else 'text_keyframes',
                                   'keyframes': m['keyframes'], 'audio_mode': 'native'}
                        expected = adapter.author._system_prompt(task, frame_count/24,
                            [item['label'] for item in m['media']], language, context, prompt)
                        self.assertEqual(messages[0]['content'], expected, (task, language, prompt))
                        count += 1
        self.assertEqual(count, 120)

    def test_multi_reference_pixels_order_roles_and_ignored_tail(self):
        task, messages, mapping = self.prepare(mode=nodes.REFERENCES, first_frame=self.image(value=0),
            last_frame=self.image(value=0.3), reference_4=self.image(value=1), reference_batch=self.image(2,value=0.5))
        self.assertEqual(task, 'Ref2VA')
        m = json.loads(mapping)
        self.assertEqual([i['input'] for i in m['media']],
                         ['first_frame','reference_4','reference_batch[0]','reference_batch[1]'])
        self.assertEqual([i['label'] for i in m['media']], [f'<picture {i}>' for i in range(1,5)])
        self.assertEqual(m['keyframes'], [])
        self.assertEqual(len(m['ignored_inputs']), 1)
        content = messages[1]['content']
        self.assertEqual(len(content), 9)
        values = []
        for part in content:
            if part['type'] == 'image_url':
                image = Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',')[1])))
                values.append(image.getpixel((0,0))[0])
        self.assertLess(values[0], 5)
        self.assertGreater(values[1], 250)
        self.assertTrue(120 < values[2] < 135)
        self.assertEqual(values[2], values[3])

    def test_mode_switch_selects_only_active_rule_and_images(self):
        image = self.image()
        _, messages, mapping = self.prepare(first_frame=image, reference_2=image,
            keyframe_rules='BASE {{task}}', reference_rules='{{invalid}}')
        self.assertEqual(messages[0]['content'], 'BASE I2VA')
        self.assertEqual(len(json.loads(mapping)['media']), 1)
        _, messages, _ = self.prepare(mode=nodes.REFERENCES, reference_2=image,
            keyframe_rules='{{invalid}}', reference_rules='REFERENCE {{task}}')
        self.assertEqual(messages[0]['content'], 'REFERENCE Ref2VA')

    def test_editable_rules_no_hidden_system_append(self):
        _, messages, _ = self.prepare(keyframe_rules='Only my own rule.', prompt='唱歌 替换')
        self.assertEqual(messages[0]['content'], 'Only my own rule.')
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]['content'][0]['text'], 'User prompt:\n唱歌 替换')

    def test_renderer_validation_and_no_recursive_evaluation(self):
        for bad in ('', '{{bad}}', '{{#singing}}unclosed', '{{/zh}}', '{{#bad}}x{{/bad}}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.prepare(keyframe_rules=bad)
        _, messages, _ = self.prepare(keyframe_rules='{{user_prompt}}',
                                      prompt='{{#zh}}literal {{task}}{{/zh}}')
        self.assertEqual(messages[0]['content'], '{{#zh}}literal {{task}}{{/zh}}')
        _, messages, _ = self.prepare(keyframe_rules='{{#zh}}A{{#singing}}B{{/singing}}{{/zh}}',prompt='唱歌')
        self.assertEqual(messages[0]['content'], 'AB')

    def test_images_and_limits_validated_before_loading(self):
        for args in ({'first_frame':self.image(2)},
                     {'mode':nodes.REFERENCES},
                     {'mode':nodes.REFERENCES,'reference_batch':self.image(10)},
                     {'mode':'unknown'}, {'fps':0}):
            with self.subTest(args=list(args)), self.assertRaises(ValueError):
                self.prepare(**args)
        _, _, mapping = self.prepare(mode=nodes.REFERENCES, reference_batch=self.image(9))
        self.assertEqual(len(json.loads(mapping)['media']), 9)

    def test_author_functions_unchanged_and_three_outputs(self):
        manifest = json.loads((ROOT/'source_reference.json').read_text())
        source = (PLUGIN/'author_rules.py').read_text()
        functions = {n.name: ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        for name, sha in manifest['upstream']['function_sha256'].items():
            self.assertEqual(hashlib.sha256(functions[name].encode()).hexdigest(), sha)
        self.assertEqual(nodes.H3AuthorGemma4ExpandV2.RETURN_NAMES,
                         ('optimized_prompt','applied_rules','media_mapping'))


class WorkflowContractTests(unittest.TestCase):


    def test_prompt_only_examples_have_no_video_execution(self):
        for path in (ROOT/'workflows').glob('*.json'):
            if path.name == '手动释放Gemma缓存.json':
                continue
            ui = json.loads(path.read_text())
            byid = {n['id']:n for n in ui['nodes']}
            self.assertEqual(sum(n['type']=='H3AuthorGemma4ExpandV2' for n in ui['nodes']),1)
            allowed = {'H3AuthorGemma4ExpandV2','ShowText|pysssss','LoadImage','JWInteger','PrimitiveStringMultiline'}
            self.assertTrue({n['type'] for n in ui['nodes']} <= allowed)
            for lid,src,ss,dst,ds,typ in ui['links']:
                self.assertIn(lid, byid[src]['outputs'][ss]['links'])
                self.assertEqual(byid[dst]['inputs'][ds]['link'], lid)


if __name__ == '__main__':
    unittest.main(verbosity=2)

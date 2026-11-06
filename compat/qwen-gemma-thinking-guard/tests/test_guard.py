import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qwen3vl_guarded_run as guard
import install

ANSWER = '{"prompt":"保留人物身份，将外套改成蓝色。"}'
NATIVE = '<|channel>thought\n检查服装与人物。\n<channel|>\n'
CONFIG = dict(chat_handler='gemma4', gemma_thinking_guard=True, enable_thinking=True,
              max_tokens=8192, n_ctx=32768, penalty_last_n=4096, stop_on_repetition=True,
              repeat_penalty=1.08, temperature=1.0, top_k=64, top_p=.95, min_p=0.0,
              system_prompt='RULES', user_prompt='USER', debug=False)


class FakeLlama:
    """Reproduces a chat handler which drops penalty_last_n and may drop final text."""
    def __init__(self, raw=NATIVE+ANSWER, finish='stop', step=7, separate=False, drop_final=False):
        self.raw, self.finish, self.step = raw, finish, step
        self.separate, self.drop_final = separate, drop_final
        self.seen, self.yielded, self.closed = [], 0, False
        self.chat_handler = object()

    def create_completion(self, prompt, stream=False, penalty_last_n=64, **kwargs):
        self.seen.append(dict(prompt=prompt, stream=stream, penalty_last_n=penalty_last_n, **kwargs))
        def chunks():
            try:
                for i in range(0, len(self.raw), self.step):
                    self.yielded += 1
                    last = i + self.step >= len(self.raw)
                    yield {'choices':[{'text':self.raw[i:i+self.step], 'finish_reason':self.finish if last else None}]}
            finally:
                self.closed = True
        if stream:
            return chunks()
        return {'choices':[{'text':self.raw,'finish_reason':self.finish}]}

    def create_chat_completion(self, messages, stream=False, **kwargs):
        kwargs.pop('penalty_last_n', None)  # Real problem: argument dropped by multimodal handler.
        if not stream:
            self.create_completion('ENCODED', **kwargs)
            return {'choices':[{'message':{'content':self.raw},'finish_reason':self.finish}]}
        def converted():
            source = self.create_completion('ENCODED', stream=True,
                stop=['<eos>','<turn|>','<|tool_response>'], **kwargs)
            try:
                for chunk in source:
                    choice=chunk['choices'][0]
                    if self.separate:
                        text=choice['text'];delta={'reasoning_content':text}
                    else:
                        delta={'content':'' if self.drop_final and choice['finish_reason'] else choice['text']}
                    yield {'choices':[{'delta':delta,'finish_reason':choice['finish_reason']}]}
                if self.separate:
                    yield {'choices':[{'delta':{'content':ANSWER},'finish_reason':'stop'}]}
            finally:
                source.close()
        return converted()


def invoke(llm, config=None):
    conf=dict(CONFIG,**(config or {}));trace=guard.Trace(conf,lambda:None)
    proxy=guard.GuardedLlama(llm,conf,trace)
    return proxy.create_chat_completion(messages=[],repeat_penalty=1.08,temperature=1.0),trace


class GuardTests(unittest.TestCase):
    def test_native_chunks_stripped_and_window_reaches_completion(self):
        llm=FakeLlama();response,trace=invoke(llm)
        self.assertEqual(response['choices'][0]['message']['content'],ANSWER)
        self.assertEqual(llm.seen[0]['penalty_last_n'],4096)
        self.assertEqual(llm.seen[0]['repeat_penalty'],1.08)
        self.assertEqual(llm.seen[0]['stop'],['<eos>','<turn|>','<|tool_response>'])
        self.assertNotIn('create_completion',vars(llm));self.assertTrue(llm.closed)
        self.assertEqual(response['usage'],{})

    def test_split_every_marker_boundary(self):
        for step in range(1,25):
            with self.subTest(step=step):
                response,_=invoke(FakeLlama(step=step))
                self.assertEqual(response['choices'][0]['message']['content'],ANSWER)

    def test_xml_thought_tags(self):
        result,_=invoke(FakeLlama('<think>检查人物。</think>'+ANSWER,step=1))
        self.assertEqual(result['choices'][0]['message']['content'],ANSWER)

    def test_valid_json_with_length_still_rejected(self):
        llm=FakeLlama(ANSWER,finish='length')
        with self.assertRaisesRegex(guard.GuardError,'OUTPUT_TRUNCATED'):invoke(llm)
        self.assertNotIn('create_completion',vars(llm));self.assertTrue(llm.closed)

    def test_missing_finish_rejected(self):
        with self.assertRaisesRegex(guard.GuardError,'INCOMPLETE_RESPONSE'):invoke(FakeLlama(ANSWER,finish=None))

    def test_unknown_finish_rejected(self):
        with self.assertRaisesRegex(guard.GuardError,'INCOMPLETE_RESPONSE'):invoke(FakeLlama(ANSWER,finish='tool_calls'))

    def test_unfinished_thinking_rejected(self):
        with self.assertRaisesRegex(guard.GuardError,'UNFINISHED_THINKING'):invoke(FakeLlama('<|channel>thought\n未结束'))

    def test_empty_final_rejected(self):
        with self.assertRaisesRegex(guard.GuardError,'EMPTY_ANSWER'):invoke(FakeLlama(NATIVE))

    def test_repeated_line_group_stops_early(self):
        block='Check character identity, clothing and exact object ownership.\nCheck each relationship and preserve the unchanged scene.\n'
        llm=FakeLlama('<|channel>thought\n'+block*100+'<channel|>'+ANSWER)
        with self.assertRaisesRegex(guard.GuardError,'THINKING_LOOP'):invoke(llm)
        self.assertLess(llm.yielded,200);self.assertTrue(llm.closed)
        self.assertNotIn('create_completion',vars(llm))

    def test_no_newline_loop_stops(self):
        block='The same long reasoning sentence about characters repeats without any newline. '
        with self.assertRaisesRegex(guard.GuardError,'THINKING_LOOP'):
            invoke(FakeLlama('<think>'+block*100+'</think>'+ANSWER))

    def test_separate_reasoning_channel_stops(self):
        thought='Check every character and verify every object ownership again.\n'*100
        with self.assertRaisesRegex(guard.GuardError,'THINKING_LOOP'):invoke(FakeLlama(thought,separate=True))

    def test_separate_reasoning_channel_success(self):
        result,_=invoke(FakeLlama('Brief check of identities.',separate=True))
        self.assertEqual(result['choices'][0]['message']['content'],ANSWER)

    def test_final_repetition_is_not_thought_loop(self):
        answer=json.dumps({'prompt':'This intentionally repeated final refrain is part of an image caption. '*20})
        result,_=invoke(FakeLlama(NATIVE+answer))
        self.assertEqual(result['choices'][0]['message']['content'],answer)

    def test_marker_literal_inside_json_preserved(self):
        answer=json.dumps({'prompt':'Print <think> and <|channel>thought on a sign.'})
        result,_=invoke(FakeLlama(answer))
        self.assertEqual(result['choices'][0]['message']['content'],answer)

    def test_detector_can_be_disabled(self):
        block='Check character identity, clothing and exact object ownership.\n'
        result,_=invoke(FakeLlama('<think>'+block*5+'</think>'+ANSWER),{'stop_on_repetition':False})
        self.assertEqual(result['choices'][0]['message']['content'],ANSWER)

    def test_last_chunk_text_kept_if_chat_handler_drops_it(self):
        result,_=invoke(FakeLlama(drop_final=True))
        self.assertEqual(result['choices'][0]['message']['content'],ANSWER)

    def test_cancellation_closes_stream_restores_instance(self):
        llm=FakeLlama();calls=[]
        def cancel():
            calls.append(1)
            if len(calls)>5:raise RuntimeError('CANCELLED')
        trace=guard.Trace(CONFIG,cancel)
        with self.assertRaisesRegex(RuntimeError,'CANCELLED'):
            guard.GuardedLlama(llm,CONFIG,trace).create_chat_completion(messages=[])
        self.assertTrue(llm.closed);self.assertNotIn('create_completion',vars(llm))

    def test_existing_instance_override_restored(self):
        llm=FakeLlama();original=llm.create_completion;llm.create_completion=original
        invoke(llm);self.assertIs(vars(llm)['create_completion'],original)

    def test_unsupported_window_fails_explicitly(self):
        llm=FakeLlama();llm.create_completion=lambda **kwargs:None
        with self.assertRaisesRegex(guard.GuardError,'BACKEND_INCOMPATIBLE'):invoke(llm)

    def test_invalid_window_rejected(self):
        for window in (0,-1,True,1.5):
            with self.subTest(window=window),self.assertRaisesRegex(guard.GuardError,'INVALID_CONFIG'):
                invoke(FakeLlama(),{'penalty_last_n':window})

    def test_previous_failure_does_not_poison_next_call(self):
        llm=FakeLlama(ANSWER,finish='length')
        with self.assertRaises(guard.GuardError):invoke(llm)
        llm.finish='stop'
        self.assertEqual(invoke(llm)[0]['choices'][0]['message']['content'],ANSWER)

    def test_runtime_keeps_success_releases_failure(self):
        for finish in ('stop','length'):
            released=[];llm=FakeLlama(finish=finish)
            def infer(proxy,config,images,audios,videos):
                self.assertFalse(config['streaming_mode'])
                result=proxy.create_chat_completion(messages=[])
                return {'status':'success','output':result['choices'][0]['message']['content']},None
            original=dict(CONFIG)
            result,_=guard.run_guarded(infer,llm,CONFIG,[],[],[],lambda:released.append(1))
            self.assertEqual(result['status'],'success' if finish=='stop' else 'error')
            self.assertEqual(released,[] if finish=='stop' else [1]);self.assertEqual(CONFIG,original)
            if finish=='length':self.assertIn('原始输出末尾',result['message'])


# AST isolation executes the real audited upstream functions, with only hardware
# and message encoding substituted. Supply paths with --source-dir? Uses env below.
import os
SOURCE_DIR=Path(os.environ.get('QWEN_GUARD_TEST_SOURCE','/tmp/qwen-guard-audit'))


@unittest.skipUnless((SOURCE_DIR/'qwen3vl_run.py').exists(),'audited upstream fixtures unavailable')
class UpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import time,traceback,re
        tree=ast.parse((SOURCE_DIR/'qwen3vl_run.py').read_text())
        names={'_infer_chat','_postprocess_output','_prepare_completion_kwargs','_parse_strings_list','_norm_str'}
        cls.ns=dict(time=time,traceback=traceback,re=re,json=json,sys=sys,
            _debug_print=lambda *a,**k:None,_build_message_content=lambda config,images,audios,videos,**kw:images+[kw],
            _build_text_prompt=lambda *args:(None,None))
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'upstream','exec'),cls.ns)

    def test_baseline_original_accepts_length_and_loses_window(self):
        llm=FakeLlama(ANSWER,finish='length')
        result,_=self.ns['_infer_chat'](llm,dict(CONFIG),['reference'],[],[])
        self.assertEqual(result['status'],'success')
        self.assertEqual(llm.seen[0]['penalty_last_n'],64)

    def test_guard_with_real_upstream_rejects_length(self):
        released=[]
        result,_=guard.run_guarded(self.ns['_infer_chat'],FakeLlama(ANSWER,finish='length'),CONFIG,['reference'],[],[],lambda:released.append(1))
        self.assertEqual(result['status'],'error');self.assertIn('OUTPUT_TRUNCATED',result['message']);self.assertEqual(released,[1])

    def test_guard_with_real_upstream_preserves_sampler(self):
        llm=FakeLlama();released=[]
        result,_=guard.run_guarded(self.ns['_infer_chat'],llm,CONFIG,['reference'],[],[],lambda:released.append(1))
        self.assertEqual(result['output'],ANSWER);self.assertFalse(released)
        for key in ('temperature','top_k','top_p','min_p','repeat_penalty','max_tokens'):
            self.assertEqual(llm.seen[0][key],CONFIG[key])
        self.assertEqual(llm.seen[0]['penalty_last_n'],4096)

    def test_installer_idempotence_verify_and_exact_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin=Path(tmp)
            for name in install.HASHES:(plugin/name).write_bytes((SOURCE_DIR/name).read_bytes())
            install.install(plugin);install.install(plugin);install.install(plugin,'verify')
            self.assertIn('_infer_chat_without_guard',(plugin/'qwen3vl_run.py').read_text())
            install.install(plugin,'rollback')
            for name in install.HASHES:self.assertEqual((plugin/name).read_bytes(),(SOURCE_DIR/name).read_bytes())
            self.assertFalse((plugin/install.HELPER).exists())
            install.install(plugin);install.install(plugin,'verify')

    def test_installer_mismatch_has_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin=Path(tmp)
            for name in install.HASHES:(plugin/name).write_bytes((SOURCE_DIR/name).read_bytes())
            target=plugin/'qwen3vl_node.py';target.write_bytes(target.read_bytes()+b'\n# local change\n')
            before={p.name:p.read_bytes() for p in plugin.iterdir()}
            with self.assertRaisesRegex(RuntimeError,'版本不匹配'):install.install(plugin)
            self.assertEqual(before,{p.name:p.read_bytes() for p in plugin.iterdir()})

    def test_installer_refuses_overwriting_post_install_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin=Path(tmp)
            for name in install.HASHES:(plugin/name).write_bytes((SOURCE_DIR/name).read_bytes())
            install.install(plugin)
            target=plugin/'qwen3vl_run.py';target.write_bytes(target.read_bytes()+b'\n# local change\n')
            with self.assertRaisesRegex(RuntimeError,'安装后文件被改动'):install.install(plugin,'rollback')

    def test_script_alias_fails_before_install_and_resolves_after(self):
        import types
        raw=(SOURCE_DIR/'qwen3vl_node.py').read_bytes()
        for installed in (False,True):
            text=install.transform('qwen3vl_node.py',raw).decode() if installed else raw.decode()
            tree=ast.parse(text);func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_inference_pipeline')
            calls=[]
            module=types.SimpleNamespace(_infer_chat_without_guard=True,
                run_inference_direct=lambda conf:(calls.append(conf) or {'status':'success','output':ANSWER,'data_type':0},None))
            ns=dict(qwen3vl_run=module,_current_module=None,sys=sys,
                    unload_model=lambda *a,**k:None,traceback=__import__('traceback'))
            exec(compile(ast.Module(body=[func],type_ignores=[]),'upstream-node','exec'),ns)
            result=ns['run_inference_pipeline']('qwen3vl_guarded_run.py',dict(CONFIG),mode='keep_vram')
            self.assertEqual(bool(calls),installed)
            self.assertEqual(result[0],ANSWER) if installed else self.assertIn('not supported',result[0])


if __name__=='__main__':unittest.main()

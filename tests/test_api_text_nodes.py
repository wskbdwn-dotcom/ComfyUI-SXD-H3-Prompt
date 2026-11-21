"""Offline checks for the API text node's successful and fallback paths."""

import json
import base64
import io
import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading
import unittest
from unittest import mock
from urllib import error

spec = importlib.util.spec_from_file_location(
    'api_text_nodes_test', Path(__file__).resolve().parents[1] / 'api_text_nodes.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body, ensure_ascii=False).encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


class APITextNodeTests(unittest.TestCase):
    def setUp(self):
        self.node = api.SXDOpenAITextFallback()
        self.params = dict(
            original_prompt=' 女人说：“帮帮我”  ', system_prompt='Keep the dialogue.',
            api_base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
            api_key='secret-key', model='qwen-plus', temperature=0.2,
            top_p=1.0, max_tokens=1024, max_tokens_field='max_tokens', timeout_sec=5,
            extra_headers_json='{"X-DashScope-DataInspection":{"input":"cip"}}',
            extra_body_json='{"stream":false}',
        )

    def test_success_sends_system_user_key_and_extra_headers(self):
        answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'H3 formatted text'}}]}
        with mock.patch.object(api, '_open', return_value=FakeResponse(answer)) as opened:
            self.assertEqual(self.node.run(**self.params), ('H3 formatted text', 'success'))
        req = opened.call_args.args[0]
        self.assertEqual(req.full_url, self.params['api_base_url'] + '/chat/completions')
        self.assertEqual(req.get_method(), 'POST')
        self.assertEqual(req.get_header('Authorization'), 'Bearer secret-key')
        self.assertEqual(req.get_header('User-agent'), 'ComfyUI-SXD-H3-Prompt/1.0')
        self.assertEqual(req.get_header('X-dashscope-datainspection'), '{"input":"cip"}')
        body = json.loads(req.data)
        self.assertEqual(body['messages'], [
            {'role': 'system', 'content': 'Keep the dialogue.'},
            {'role': 'user', 'content': self.params['original_prompt']}])
        self.assertEqual(body['stream'], False)
        self.assertEqual(opened.call_args.args[1], 5)

    def test_full_endpoint_and_custom_authorization(self):
        params = dict(self.params, api_base_url=self.params['api_base_url'] + '/chat/completions',
                      extra_headers_json='{"Authorization":"Custom token"}')
        answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'result'}}]}
        with mock.patch.object(api, '_open', return_value=FakeResponse(answer)) as opened:
            self.assertEqual(self.node.run(**params), ('result', 'success'))
        self.assertEqual(opened.call_args.args[0].full_url, params['api_base_url'])
        self.assertEqual(opened.call_args.args[0].get_header('Authorization'), 'Custom token')

    def test_alternate_token_field(self):
        params = dict(self.params, max_tokens_field='max_completion_tokens')
        answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'result'}}]}
        with mock.patch.object(api, '_open', return_value=FakeResponse(answer)) as opened:
            self.assertEqual(self.node.run(**params), ('result', 'success'))
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual(body['max_completion_tokens'], 1024)
        self.assertNotIn('max_tokens', body)

    def test_optional_seed_controls_cache_without_changing_api_request(self):
        self.assertIn('seed', self.node.INPUT_TYPES()['optional'])
        answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'result'}}]}
        with mock.patch.object(api, '_open', return_value=FakeResponse(answer)) as opened:
            self.assertEqual(self.node.run(**dict(self.params, seed=1)), ('result', 'success'))
            first_request = opened.call_args.args[0]
            self.assertEqual(self.node.run(**dict(self.params, seed=2)), ('result', 'success'))
            second_request = opened.call_args.args[0]
        self.assertEqual(first_request.data, second_request.data)
        self.assertNotIn('seed', json.loads(second_request.data))

    def test_http_error_returns_exact_original_and_no_secret(self):
        failure = error.HTTPError('https://secret.invalid', 401, 'secret-key', {}, None)
        with mock.patch.object(api, '_open', side_effect=failure):
            text, status = self.node.run(**self.params)
        self.assertEqual(text, self.params['original_prompt'])
        self.assertEqual(status, 'fallback: http_401')
        self.assertNotIn('secret-key', status)

    def test_timeout_and_network_error_return_original(self):
        for failure, expected in ((TimeoutError(), 'timeout'),
                                  (error.URLError(TimeoutError()), 'timeout'),
                                  (error.URLError('private endpoint'), 'network_error')):
            with self.subTest(expected=expected):
                with mock.patch.object(api, '_open', side_effect=failure):
                    self.assertEqual(self.node.run(**self.params),
                                     (self.params['original_prompt'], 'fallback: ' + expected))

    def test_incomplete_empty_and_malformed_responses_return_original(self):
        cases = [
            ({'choices': [{'finish_reason': 'length', 'message': {'content': 'partial'}}]}, 'incomplete_response'),
            ({'choices': [{'message': {'content': 'unmarked'}}]}, 'incomplete_response'),
            ({'choices': [{'finish_reason': 'stop', 'message': {'content': ''}}]}, 'empty_response'),
            ({'error': {'message': 'secret'}}, 'invalid_response'),
            ({'choices': []}, 'invalid_response'),
        ]
        for answer, reason in cases:
            with self.subTest(reason=reason):
                with mock.patch.object(api, '_open', return_value=FakeResponse(answer)):
                    self.assertEqual(self.node.run(**self.params),
                                     (self.params['original_prompt'], 'fallback: ' + reason))

    def test_invalid_headers_and_body_do_not_send_request(self):
        cases = [(dict(extra_headers_json='{"X-Test":"bad\\nline"}'), 'invalid_headers'),
                 (dict(extra_body_json='{"messages":[]}'), 'invalid_body'),
                 (dict(extra_body_json='{"stream":true}'), 'invalid_body'),
                 (dict(extra_headers_json='{bad'), 'invalid_json')]
        with mock.patch.object(api, '_open') as opened:
            for changes, reason in cases:
                with self.subTest(reason=reason):
                    self.assertEqual(self.node.run(**dict(self.params, **changes)),
                                     (self.params['original_prompt'], 'fallback: ' + reason))
        opened.assert_not_called()

    def test_redirect_does_not_forward_api_key(self):
        received = []

        class Destination(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(self.headers.get('Authorization'))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_):
                pass

        class Source(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header('Location', destination_url)
                self.end_headers()

            def log_message(self, *_):
                pass

        destination = HTTPServer(('127.0.0.1', 0), Destination)
        destination_url = 'http://127.0.0.1:{}/redirected'.format(destination.server_port)
        source = HTTPServer(('127.0.0.1', 0), Source)
        threads = [threading.Thread(target=server.serve_forever, daemon=True)
                   for server in (source, destination)]
        for thread in threads:
            thread.start()
        try:
            params = dict(self.params, api_base_url='http://127.0.0.1:{}/v1'.format(source.server_port))
            self.assertEqual(self.node.run(**params),
                             (self.params['original_prompt'], 'fallback: http_302'))
            self.assertEqual(received, [])
        finally:
            for server in (source, destination):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=2)

    def test_unexpected_programming_error_is_not_hidden_as_api_fallback(self):
        with mock.patch.object(api, '_open', side_effect=RuntimeError('unexpected')):
            with self.assertRaisesRegex(RuntimeError, 'unexpected'):
                self.node.run(**self.params)

    def test_dynamic_image_schema_keeps_seed_and_checks_image_type(self):
        optional = self.node.INPUT_TYPES()['optional']
        self.assertIn('seed', optional)
        self.assertIn('image_10', optional)
        self.assertEqual(optional['image_10'], ('IMAGE',))
        self.assertNotIn('image_0', optional)
        self.assertNotIn('unrelated', optional)
        self.assertEqual(set(json.loads(json.dumps(optional))), {'seed', 'image_1'})

    def test_multi_image_request_preserves_numeric_slot_batch_order_and_sizes(self):
        import numpy as np
        from PIL import Image

        # Mixed dimensions must not require a resizing/concatenation node.
        inputs = {'image_10': np.ones((1, 4, 3, 4), dtype=np.float32),
                  'image_2': np.zeros((2, 2, 5, 3), dtype=np.float32), 'image_1': None}
        inputs['image_2'][1] = 0.5
        answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"people":2}'}}]}
        with mock.patch.object(api, '_open', return_value=FakeResponse(answer)) as opened:
            self.assertEqual(self.node.run(**self.params, **inputs), ('{"people":2}', 'success'))
        opened.assert_called_once()
        body = json.loads(opened.call_args.args[0].data)
        content = body['messages'][-1]['content']
        self.assertEqual(content[0], {'type': 'text', 'text': self.params['original_prompt']})
        self.assertEqual([item['text'] for item in content[1::2]], [
            '<Image 1> (image_2, item 1)', '<Image 2> (image_2, item 2)',
            '<Image 3> (image_10, item 1)'])
        decoded = [Image.open(io.BytesIO(base64.b64decode(item['image_url']['url'].split(',', 1)[1])))
                   for item in content[2::2]]
        self.assertEqual([image.size for image in decoded], [(5, 2), (5, 2), (3, 4)])
        self.assertEqual(decoded[0].getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(decoded[1].getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(decoded[2].mode, 'RGBA')

    def test_image_tensor_is_detached_and_moved_to_cpu(self):
        import numpy as np

        frame = mock.Mock()
        frame.detach.return_value.cpu.return_value.float.return_value.numpy.return_value = np.zeros((2, 3, 1))
        batch = mock.Mock(ndim=4, shape=(1, 2, 3, 1))
        batch.__iter__ = mock.Mock(return_value=iter([frame]))
        content = api._user_content('look', {'image_1': batch})
        self.assertEqual(len(content), 3)
        frame.detach.return_value.cpu.assert_called_once()

    def test_invalid_images_fall_back_before_request(self):
        import numpy as np

        for batch in (np.zeros((2, 2, 3)), np.zeros((1, 2, 2, 2)),
                      np.full((1, 2, 2, 3), np.nan), np.zeros((1, 0, 2, 3))):
            with self.subTest(shape=batch.shape):
                with mock.patch.object(api, '_open') as opened:
                    self.assertEqual(self.node.run(**self.params, image_1=batch),
                                     (self.params['original_prompt'], 'fallback: invalid_images'))
                opened.assert_not_called()

    def test_empty_and_unconnected_image_inputs_preserve_text_request(self):
        import numpy as np

        self.assertEqual(api._user_content(' unchanged ', {'image_1': None}), ' unchanged ')
        self.assertEqual(api._user_content(' unchanged ', {'image_1': np.zeros((0, 2, 2, 3))}), ' unchanged ')

    def test_explicit_json_mode_validates_and_normalizes_object(self):
        params = dict(self.params, extra_body_json='{"response_format":{"type":"json_object"}}')
        answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': ' { "ready": true } '}}]}
        with mock.patch.object(api, '_open', return_value=FakeResponse(answer)) as opened:
            self.assertEqual(self.node.run(**params), ('{"ready":true}', 'success'))
        self.assertEqual(json.loads(opened.call_args.args[0].data)['response_format'], {'type': 'json_object'})

    def test_bad_json_mode_response_returns_exact_original(self):
        params = dict(self.params, extra_body_json='{"response_format":{"type":"json_object"}}')
        for text in ('plain text', '[]', '```json\n{}\n```', '{"a":1,"a":2}', '{"a":NaN}'):
            answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}
            with self.subTest(text=text):
                with mock.patch.object(api, '_open', return_value=FakeResponse(answer)):
                    self.assertEqual(self.node.run(**params),
                                     (self.params['original_prompt'], 'fallback: invalid_response_json'))

    def test_normal_text_mode_does_not_require_json(self):
        self.assertEqual(api._content({'choices': [{'finish_reason': 'stop', 'message': {'content': 'plain text'}}]}),
                         'plain text')

if __name__ == '__main__':
    unittest.main()

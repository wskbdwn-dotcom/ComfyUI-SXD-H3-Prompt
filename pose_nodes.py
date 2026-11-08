# SPDX-License-Identifier: GPL-3.0-or-later
"""Program node: turn a contact record into one pose, or stop the branch."""
from . import pose_route

_ROUTES_READY = False


def _blocker(message):
    try:
        from comfy_execution.graph import ExecutionBlocker
    except Exception:
        raise RuntimeError(message)
    return ExecutionBlocker(message)


class SXDPoseRoute:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'contact_json': ('STRING', {'forceInput': True}),
            'original_request': ('STRING', {'forceInput': True}),
            'general_rules': ('STRING', {'forceInput': True}),
            'pose_rules': ('STRING', {'forceInput': True}),
        }}

    RETURN_TYPES = ('STRING', 'STRING', 'STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('system_prompt', 'user_prompt', 'status', 'extraction', 'matches')
    FUNCTION = 'run'
    CATEGORY = 'SXD/Gemma Image'
    DESCRIPTION = '按接触记录从已上传的姿势库取一条。匹配列表和提取结果始终输出；没有可选取条目就拦住后面的扩写。'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return pose_route.library_token(pose_route.STORE)

    def run(self, contact_json, original_request, general_rules, pose_rules):
        result = pose_route.prepare(
            contact_json, original_request, general_rules, pose_rules, pose_route.STORE)
        if result['ok']:
            system, user = result['system'], result['user']
        else:
            system = user = _blocker(result['status'])
        return system, user, result['status'], result['extraction'], result['matches']


def register_pose_routes():
    global _ROUTES_READY
    if _ROUTES_READY:
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    server = getattr(PromptServer, 'instance', None)
    if server is None:
        return

    @server.routes.post('/sxd/pose-library')
    async def upload_pose_library(request):
        form = await request.post()
        field = form.get('file')
        blob = getattr(field, 'file', None)
        if blob is None:
            return web.json_response({'error': '没有上传文件。'}, status=400)
        try:
            count = pose_route.install_zip(blob.read())
        except ValueError as error:
            return web.json_response({'error': str(error)}, status=400)
        return web.json_response({'count': count, 'dir': str(pose_route.STORE)})

    _ROUTES_READY = True

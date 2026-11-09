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


def load_pose_image(path):
    """Load one installed pose image as a ComfyUI IMAGE tensor."""
    import numpy as np
    import torch
    from PIL import Image
    picture = Image.open(path).convert('RGB')
    array = np.asarray(picture, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError('姿势图无法作为参考图2：' + str(path))
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


class SXDPoseRoute:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'contact_json': ('STRING', {'forceInput': True}),
            'original_request': ('STRING', {'forceInput': True}),
            'general_rules': ('STRING', {'forceInput': True}),
            'pose_rules': ('STRING', {'forceInput': True}),
            'image_json': ('STRING', {'forceInput': True}),
        }, 'optional': {
            'image2': ('IMAGE',),
        }}

    RETURN_TYPES = ('STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'STRING', 'IMAGE')
    RETURN_NAMES = ('system_prompt', 'user_prompt', 'status', 'extraction', 'matches',
                    'support_choice', 'image_state', 'pose_image')
    FUNCTION = 'run'
    CATEGORY = 'SXD/Gemma Image'
    DESCRIPTION = '按接触记录从已上传的姿势库取一条。取到姿势图时从 pose_image 输出，接到生图的参考图2。没有可选取条目就拦住后面的扩写。'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return pose_route.library_token(pose_route.STORE)

    def run(self, contact_json, original_request, general_rules, pose_rules, image_json, image2=None):
        result = pose_route.prepare(
            contact_json, original_request, general_rules, pose_rules, pose_route.STORE, image_json)
        if result['ok']:
            system, user = result['system'], result['user']
        else:
            system = user = _blocker(result['status'])
        chosen = result.get('pose_image') or ''
        pose_image = load_pose_image(chosen) if chosen else image2
        return (system, user, result['status'], result['extraction'], result['matches'],
                result['support'], result['image_state'], pose_image)


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

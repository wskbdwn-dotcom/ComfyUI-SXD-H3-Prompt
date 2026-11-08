# SPDX-License-Identifier: GPL-3.0-or-later
from .nodes import H3AuthorGemma4ExpandV2
from .runtime import H3AuthorGemma4Release
NODE_CLASS_MAPPINGS = {"H3AuthorGemma4ExpandV2": H3AuthorGemma4ExpandV2, "H3AuthorGemma4ReleaseV2": H3AuthorGemma4Release}
NODE_DISPLAY_NAME_MAPPINGS = {"H3AuthorGemma4ExpandV2": "H3 Gemma4 V2 · 可编辑规则 / 首尾帧 / 多图参考", "H3AuthorGemma4ReleaseV2": "H3 Gemma4 V2 · 手动释放常驻模型"}

from .live import H3PromptLiveDisplay
NODE_CLASS_MAPPINGS['H3PromptLiveDisplay'] = H3PromptLiveDisplay
NODE_DISPLAY_NAME_MAPPINGS['H3PromptLiveDisplay'] = 'H3 · 实时输出查看（思考 / 正文）'
WEB_DIRECTORY = './web'

from .image_nodes import SXDGemmaImageTask
NODE_CLASS_MAPPINGS['SXDGemmaImageTask'] = SXDGemmaImageTask
NODE_DISPLAY_NAME_MAPPINGS['SXDGemmaImageTask'] = 'SXD Gemma · 图像扩写 / 大结构检查'

from .image_nodes import SXDGemmaSpatialPlan, SXDSpatialPlanAssemble
NODE_CLASS_MAPPINGS['SXDGemmaSpatialPlan'] = SXDGemmaSpatialPlan
NODE_CLASS_MAPPINGS['SXDSpatialPlanAssemble'] = SXDSpatialPlanAssemble
NODE_DISPLAY_NAME_MAPPINGS['SXDGemmaSpatialPlan'] = 'SXD Gemma · 单图结构化规划'
NODE_DISPLAY_NAME_MAPPINGS['SXDSpatialPlanAssemble'] = 'SXD Gemma · 校验并组装编辑指令'

from .pose_nodes import SXDPoseRoute, register_pose_routes
NODE_CLASS_MAPPINGS['SXDPoseRoute'] = SXDPoseRoute
NODE_DISPLAY_NAME_MAPPINGS['SXDPoseRoute'] = 'SXD · 姿势查库'
register_pose_routes()

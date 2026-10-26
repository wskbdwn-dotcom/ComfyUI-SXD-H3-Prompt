# SPDX-License-Identifier: GPL-3.0-or-later
from .nodes import H3AuthorGemma4ExpandV2
from .runtime import H3AuthorGemma4Release
NODE_CLASS_MAPPINGS = {"H3AuthorGemma4ExpandV2": H3AuthorGemma4ExpandV2, "H3AuthorGemma4ReleaseV2": H3AuthorGemma4Release}
NODE_DISPLAY_NAME_MAPPINGS = {"H3AuthorGemma4ExpandV2": "H3 Gemma4 V2 · 可编辑规则 / 首尾帧 / 多图参考", "H3AuthorGemma4ReleaseV2": "H3 Gemma4 V2 · 手动释放常驻模型"}

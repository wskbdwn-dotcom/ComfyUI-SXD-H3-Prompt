# ComfyUI-SXD-H3-Prompt

MiniMax H3 本地提示词优化与扩写节点。支持可编辑规则、首尾帧和图片参考模式，目前推理后端支持 **Gemma4 GGUF 视觉模型**。

这是基于 Goohai H3 提示词规则的独立衍生适配，不是 MiniMax 或 Goohai 官方发布。仓库名称不限制未来的模型支持，但当前不能通过更换文件名直接运行 Qwen 等其他模型。

## 功能

- 两套可编辑规则：首尾帧规则、全能参考规则；切换模式时使用对应规则。
- 首尾帧模式根据图片输入选择 T2VA、I2VA、L2VA、FL2VA。
- 图片参考模式支持 1–9 张图片，单张图也可用于人物参考；暂不支持视频或音频参考输入。
- 中文／英文输出、Gemma4 思考开关、作者格式后处理开关。
- 默认保留 Gemma 和视觉投影器，支持扩写后释放及手动释放。
- 不主动调用 ComfyUI 的全局模型卸载；H3 是否被其他节点或显存管理卸载，仍由实际环境决定。
- 输出扩写文本、实际应用的规则和图片用途映射，便于检查。

**本节点仅扩写提示词。** 切换到参考模式不会自动把下游首帧生成工作流改成主体参考生成；H3 的模型和图片条件需要与生成任务匹配。

## 安装

在 ComfyUI 的 `custom_nodes` 目录执行：

```bash
git clone https://github.com/wskbdwn-dotcom/ComfyUI-SXD-H3-Prompt.git
```

重启 ComfyUI，在节点搜索中查找 `H3 Gemma4 V2`。

### 避免重复安装

为兼容已有工作流，内部节点类型仍是 `H3AuthorGemma4ExpandV2` 和 `H3AuthorGemma4ReleaseV2`。
如果此前已经安装 `ComfyUI-H3-Author-Gemma4-V2`，先在服务停止时把旧插件备份到 **custom_nodes 目录以外**，再安装本仓库，避免同名节点重复注册。此步骤只针对旧版同一扩写插件，不是要求移动 T8。

### 运行依赖

- 已能正常运行的 ComfyUI 环境，包含 numpy、Pillow。
- 支持 `Gemma4ChatHandler` 和显式 `enable_thinking` 参数的 [JamePeng/llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) GPU 构建。
- GGUF 主模型及与之匹配的视觉投影器（mmproj）。

此前使用的后端版本为 0.3.35；仅版本号相同不代表构建、CUDA 或模型支持相同。节点安装不会自动升级 PyTorch、CUDA 或 llama-cpp-python。`requirements.txt` 只列出普通 Python 依赖，不负责安装 GPU 后端。

将模型放在 `ComfyUI/models/llm/` 或 `ComfyUI/models/LLM/`，在节点中填写相对文件名。默认示例：

```text
gemma-4-26b-a4b-it-heretic.q4_k_m.gguf
gemma-4-26B-A4B-it-heretic-mmproj.bf16.gguf
```

本仓库不附带、不下载模型。避免在两个搜索目录放置同名的不同文件，否则节点会报匹配不唯一。

RunPod 镜像含 T8 且出现 `undefined symbol: llama_params_fit` 时，按 [RunPod 库路径说明](docs/runpod.md) 配置启动环境。

## 快速验证

将 `workflows/单首帧.json` 拖入 ComfyUI，选择自己的图片、填写提示词后运行。

示例依赖 ComfyUI 的 `LoadImage`、`PrimitiveStringMultiline`，以及额外的 `JWInteger` 和 `ShowText|pysssss`（ComfyUI-Custom-Scripts）节点。缺少时需安装对应节点，或用同类型整数输入、文本显示节点替换；它们不是扩写后端自身的依赖。

保持 `release_after_expansion=false` 连续运行两次，第二次修改提示词：

1. 首轮日志应出现 `LOAD resident model and projector`，并输出扩写文本。
2. 第二轮应出现 `REUSE resident model and projector`，并再次输出文本。

完全相同的输入可能命中 ComfyUI 结果缓存，不执行扩写。服务重启后的首次请求需要重新加载模型。

另有 `首尾帧.json`、`多图参考.json` 和 `手动释放Gemma缓存.json`。这些示例不生成视频。

## 参数与模式

| 参数 | 用途 |
|---|---|
| `mode` | 首尾帧 / 全能参考（多图） |
| `keyframe_rules` | 首尾帧模式使用的规则 |
| `reference_rules` | 参考模式使用的规则 |
| `enable_thinking` | 默认关闭；直接传给 Gemma4 处理器，不修改第三方源码 |
| `output_language` | 默认规则中的中文／英文分支；自定义规则需保留相应语言条件 |
| `author_postprocess` | 默认开启作者格式整理；完全自定义输出格式时可关闭 |
| `release_after_expansion` | 默认关闭，正常完成后保留模型；开启则扩写后释放 |

首尾帧模式：仅 `first_frame` 为 I2VA；仅 `last_frame` 为 L2VA；两者都有为 FL2VA；都没有为 T2VA。额外参考输入在此模式中被忽略。

参考模式：`first_frame` 作为普通参考图，并非指定首帧；之后接 `reference_2` 至 `reference_9`，也可接 `reference_batch`，总计最多 9 张。`last_frame` 在参考模式中被忽略。独立图片端口各接一张，批次使用 `reference_batch`。

规则支持 `{{task}}`、`{{duration}}`、`{{labels}}`、`{{keyframe_roles}}`、`{{user_prompt}}`；条件块包括 `{{#zh}}…{{/zh}}`、`en`、`FL2VA`、`not_FL2VA`、`singing`、`transfer`。保留标签配对，修改后的规则随工作流保存，不写回仓库默认文件。

默认规则在作者结构基础上渲染；作者后处理函数保留在 `author_rules.py`。全部实际参与的图片都会传给视觉模型，以最长边 512、JPEG 质量 42 预处理。

输出顺序保持兼容：`optimized_prompt`（0）、`applied_rules`（1）、`media_mapping`（2）。与既有工作流集成时，把输出 0 接到原来的提示词输入。

## 模型保留与更新

保留权重不代表保留上一轮对话。本节点在每轮独立请求前后清理推理状态；包含针对旧后端 `reset()` 不清理原生状态的适配。

改变主模型、投影器、思考开关、上下文大小或 GPU 层数会重新加载。推理异常或执行中取消会清理缓存。手动释放只作用于本插件缓存，不卸载 H3。

后续更新：

```bash
cd /comfyui/custom_nodes/ComfyUI-SXD-H3-Prompt
git pull --ff-only
```

任务完成后重启 ComfyUI，并刷新页面。非 `/comfyui` 安装请替换路径。公共镜像建议固定经验证的提交，不在每次启动时自动更新代码。

## 验证范围

```bash
python tests/test_v2.py
```

测试需要 numpy、Pillow，通过模拟 ComfyUI 和推理接口验证规则、图片预处理、缓存生命周期、思考开关及示例连线，不加载真实模型。

发布前仍需在目标 GPU 上验证连续两轮扩写，以及与 H3、其他插件共存。离线测试不等于 GPU 验收。

## 来源与许可

作者规则来自 [Goohai-MiniMax-H3_Integration](https://github.com/goohai/Goohai-MiniMax-H3_Integration)，固定提交 `2ae83e4f0f7781604e9c0ad9ec2348b0f1e30c21` 的 `prompt_optimizer.py`。

本衍生适配将规则拆成可编辑模板，加入独立 ComfyUI 节点、本地 Gemma4 后端、模型保留控制和旧后端兼容处理。七个沿用函数的源码哈希见 `source_reference.json`。

遵循 **GPL-3.0-or-later**，保留原版权声明。详见 `LICENSE` 和 `COPYING`。模型本身的许可证由模型提供方另行规定。

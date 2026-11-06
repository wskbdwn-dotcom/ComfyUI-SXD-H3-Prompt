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
- 新增实时输出查看节点，可看到原始思考和正文；取消或报错后保留已收到的预览。

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

输出顺序保持兼容：`optimized_prompt`（0）、`applied_rules`（1）、`media_mapping`（2），末尾追加 `raw_output`（3，包含原始思考与正文）。与既有工作流集成时，把输出 0 接到原来的提示词输入。

## 实时查看思考过程

更新代码后重启 ComfyUI，并强制刷新浏览器。在节点搜索中添加 **H3 · 实时输出查看（思考 / 正文）**，把扩写节点的第 **4** 个输出 `raw_output` **直接连接**到它的 `raw_output` 输入。现有前三个输出及所有参数不变；旧工作流加载时自动补上第4个端口。现有首帧连线保持不变。

```text
扩写节点 optimized_prompt（第1个输出） → 原 H3 提示词输入
扩写节点 raw_output（第4个输出）       → H3 · 实时输出查看
```

- 显示节点接收 WebSocket 推送，生成中即可阅读，不必等下游节点执行。普通文本显示节点只能在扩写成功后显示全文。
- 看到 `<|channel>thought` 是进入思考，`<channel|>` 后是正文。原始输出保留这些标记，禁止把 `raw_output` 用作 H3 提示词。
- 面板显示加载、生成、完成、报错/取消状态及后端的 `finish_reason`。结束符一般已被后端吃掉；预览中没有 `<turn|>` **不等于**未正常停止。
- 达到上限仍然报错并阻止不完整正文传给 H3；面板保留报错前的片段。刷新页面后部分预览不保留。
- 为避免浏览器卡顿，预览最多保留末尾200000字符并有提示；成功后的 `raw_output` 字符串仍是全文。显示节点执行后，完整文本也会进入 ComfyUI 执行历史；不要分享含敏感内容的执行历史。
- 预览仅发送给提交该任务的客户端；API 未提供 `client_id` 时不广播，成功后可读取显示节点的历史输出。当前实时连线定位仅支持主画布上直接连接，暂不支持中间转接节点/子图。
- 命中 ComfyUI 缓存时不重新生成，显示已有结果；需重新测试时修改 seed 或提示词。模型加载/图片编码期间尚无生成文本，短暂空白是正常的。

本次核对 [JamePeng 0.3.35 源码](https://github.com/JamePeng/llama-cpp-python/blob/232092e32b3563159a86aacb168da06c4937192b/llama_cpp/llama_chat_format.py)：处理器已使用 `<eos>`、`<turn|>`、`<|tool_response>` 停止，没有发现需纠正的停止符。插件保留后端停止条件，不把思考结束符 `<channel|>` 当作生成结束。结束标记和思考开关保持原样；实时预览本身不会修复思考循环。

流式后端若未提供 token 用量，截断诊断会标“未提供”，不会把字符或片段数量冒充 token 数量。

## 遇到重复思考时

新增采样参数放在原有控件末尾，不移动旧参数、规则框和输出端口。新默认值也适用于旧 API 请求未传这些参数的情况；这会改变生成结果，但不改变输入接口和规则。更新并重启 ComfyUI、强制刷新网页后，请检查以下控件：

| 参数 | 默认 / 首轮对照值 | 用途 |
|---|---|---|
| `temperature` | 1.0 | 替代此前固定的0.2 |
| `top_k` | 64 | 候选 token 范围 |
| `top_p` | 0.95 | 替代此前固定的0.9 |
| `min_p` | 0.0 | 关闭最小概率过滤 |
| `repeat_penalty` | 1.08 | 温和重复惩罚，1.0关闭；不保证改善所有任务 |
| `penalty_last_n` | 4096 | 重复惩罚检查的最近 token 范围 |
| `stop_on_repetition` | true | 思考重复循环时提前报错，避免持续浪费生成额度 |

前四项依据[该模型量化作者列出的采样基线](https://huggingface.co/nohurry/gemma-4-26B-A4B-it-heretic-GUFF#usage)；后两项采样值是[社区12B模型测试经验](https://discuss.ai.google.dev/t/gemma4-how-to-prevent-looping-output-in-thinking/173205)的实验起点，并非26B Heretic已验证的修复。

保持 `enable_thinking=true`、原图片与原提示词，先用 `max_tokens=8192`、`n_ctx=32768` 对照测试；这只是本轮测试设置，不降低插件上限，也不修改已保存的预算。先用相同seed对比参数影响；若仍循环，可换seed验证是否稳定复现。无需提高上下文到模型训练上限。

源码核对发现0.3.35的多模态处理器未把 `penalty_last_n` 转交给完成接口。本插件在自有模型实例调用完成接口时补传此参数，不修改第三方源码；若后端不支持，明确报错。改变这些采样参数不重载权重。

循环保护只在看到原生思考开始标记、尚未看到 `<channel|>` 时生效：检测1–12行组成的文本组，连续4遍完全相同且累计至少200字符（忽略空行及空白差异），提前报错并保留实时预览。它不会强制结束思考或续写正文，不处理所有形式的语义循环；如果任务确实需要重复检查，可关闭。出现该错误后仍会按现有异常流程释放 Gemma。

当前未实现独立思考预算。若本轮调整仍稳定重复，需要进一步做模型版本/量化和规则复杂度的对照测试，不能保证仅靠采样解决。规则内容、首帧输入、原有最终正文检查均未修改。

## 模型保留与更新

### 开启思考时遇到截断

`max_tokens` 和 `n_ctx` 的界面上限均为 1,048,576，便于在工作流中自行调整，不再受较低的插件上限限制。默认值仍分别为 4096 和 16384，既有工作流参数不会自动改变。

这个界面范围不表示模型、后端或显存支持同样的容量，也不启用上下文扩展。思考模式可先尝试 `enable_thinking=true`、`max_tokens=16384`、`n_ctx=32768`，再根据实际用量调整。输出额度包含思考及最终正文，规则、图片和输入文本还需占用上下文；增大上限并不保证输出完整。

运行日志会打印本轮思考开关和预算。若后端返回 `finish_reason=length`，错误会附带请求参数及后端提供的 token 用量（缺失时标为“未提供”），继续阻止不完整文本进入 H3。不会自动关闭思考或重试。增大上下文或切换思考会重载模型，截断异常后的 RELEASE 是清理行为。

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
python -m unittest discover -s tests -v
node tests/test_live.mjs
```

测试需要 numpy、Pillow，通过模拟 ComfyUI 和推理接口验证规则、图片预处理、缓存生命周期、思考开关及示例连线，不加载真实模型。实时输出另有浏览器事件模拟测试；仍需在实际 ComfyUI/RunPod 上验证前端显示和 GPU 推理。

发布前仍需在目标 GPU 上验证连续两轮扩写，以及与 H3、其他插件共存。离线测试不等于 GPU 验收。

## Qwen 图生图的原生 Gemma 扩写与检查

新增 **SXD Gemma · 图像扩写 / 大结构检查** 节点，直接使用本插件的 Gemma 后端，支持规则输入、1–4 张图片扩写、单张成图检查、思考保护和模型驻留开关。无需 KLL 插件或外部推理服务。相同模型配置下与 H3 节点共用一份模型缓存；原有 H3 接口、规则和默认行为保持不变。

此前的 KLL 补丁目录和安装器已移除，不再运行 `compat/.../install.py`。更新后使用配套 **v6 原生节点工作流**；v4/v5 仍引用旧节点，不会自动迁移。安装、参数和兼容范围见[图像节点说明](docs/image-tasks.md)。

### 自定义规则扩写（v9-live）

需要自己编辑扩写规则时，导入 [v9-live 自定义规则工作流](workflows/qwen/image2image_qwen2.1-gemma-v9-live-custom-rules.json)。节点 **1201** 是可编辑的规则，已预填英文规则、要求中文输出；节点 **1200** 填原始编辑要求，也可以附上已核验的图像事实。节点 **1109** 填参考图 URL。规则另存为 [custom-image-edit-rules.txt](workflows/qwen/custom-image-edit-rules.txt)，修改画布里的规则即可生效，不必改插件源码。

节点 **1151** 使用现有 `SXDGemmaImageTask` 的 `prompt` 任务，按自定义规则直接输出 `{"prompt":"…"}`，经 **1213** 显示后进入原有提取与生图链路。规则仍须要求这个输出格式。实时思考与原始输出继续显示；该流程不经过结构化规划和组装节点，也没有结构化自动纠正。原有模型参数、成图检查和按需修复链路保留。

更新后需要导入这个工作流；`git pull` 不会自动替换已经打开的画布。公开模板的规则已填好，图片地址和编辑要求留给使用者填写。旧结构化工作流仍可打开，属于下面的可选实验方式。

### 单图结构化编辑扩写（可选实验功能）

新增 **SXD Gemma · 单图结构化规划** 和 **SXD Gemma · 校验并组装编辑指令**。Gemma 生成包含人物位置、基础姿态、躯干朝向、接触端点和镜头信息的规划，程序检查后组装为 `{"prompt":"…"}`。校验失败最多自动纠正两次，仍不通过才报错；不会猜测缺失的基础姿态，组装阶段不调用模型。

基于 v9-live 的完整副本：[image2image_qwen2.1-gemma-v9-live-structured.json](workflows/qwen/image2image_qwen2.1-gemma-v9-live-structured.json)。更新插件并重启 ComfyUI、刷新页面后导入。填写节点 **1109** 的单张参考图 URL、节点 **1200** 的原始编辑要求；节点 **1201** 是可选的已核验图像事实，换图后需要重新确认。字段规则内置在插件中，此处不粘贴旧扩写规则。模型配置支持 `models/LLM` 或 `models/llm` 下的相对文件名，也可以填现有绝对路径。

副本保留实时输出、首次生图、成图检查和按需修复链路。使用原 v9-live 所需的其他节点；两次实时预览仍直接连接各自的原始输出。结构化扩写只接受单张参考图；旧扩写和检查节点继续支持原接口。

这是生成后的结构与引用检查，不是底层强制解码，也不保证图像识别、空间合理性或动作语义正确。离线测试使用模拟后端，仍需实际 Gemma 与生图验证。详细输入和边界见[结构化图像编辑说明](docs/structured-image-edit.md)。

## 来源与许可

作者规则来自 [Goohai-MiniMax-H3_Integration](https://github.com/goohai/Goohai-MiniMax-H3_Integration)，固定提交 `2ae83e4f0f7781604e9c0ad9ec2348b0f1e30c21` 的 `prompt_optimizer.py`。

本衍生适配将规则拆成可编辑模板，加入独立 ComfyUI 节点、本地 Gemma4 后端、模型保留控制和旧后端兼容处理。七个沿用函数的源码哈希见 `source_reference.json`。

遵循 **GPL-3.0-or-later**，保留原版权声明。详见 `LICENSE` 和 `COPYING`。模型本身的许可证由模型提供方另行规定。

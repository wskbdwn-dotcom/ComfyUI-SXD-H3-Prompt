# Qwen 图生图工作流的 Gemma 思考保护

为使用 KLL `SimpleQwenVLggufV2` 节点的工作流提供可选兼容补丁。它与本仓库的 H3 扩写节点分开：仅拉取本仓库不会修改 KLL 插件，必须显式运行安装命令。

补丁在 KLL 的两个文件中增加接入代码，另安装一个辅助文件；不注册新节点、不启动额外服务、不升级推理依赖。原有 H3 节点代码和接口不变。

## 服务器安装

先停止 ComfyUI。已经安装本仓库时：

```bash
cd /comfyui/custom_nodes/ComfyUI-SXD-H3-Prompt
git pull --ff-only
python3 compat/qwen-gemma-thinking-guard/install.py --plugin /comfyui/custom_nodes/ComfyUI_Simple_Qwen3-VL-gguf
```

尚未安装本仓库时，在 `/comfyui/custom_nodes` 中执行：

```bash
git clone https://github.com/wskbdwn-dotcom/ComfyUI-SXD-H3-Prompt.git
python3 ComfyUI-SXD-H3-Prompt/compat/qwen-gemma-thinking-guard/install.py --plugin /comfyui/custom_nodes/ComfyUI_Simple_Qwen3-VL-gguf
```

替换实际安装目录；`--plugin` 应指向包含 `qwen3vl_run.py` 和 `qwen3vl_node.py` 的目录。
如果已安装旧名 `ComfyUI-H3-Author-Gemma4-V2`，按主 README 的“避免重复安装”说明迁移，避免同时注册同一套 H3 节点。

安装成功后重启 ComfyUI，导入已提供的 v4 工作流。此代码目录不包含个人工作流、图片、提示词或模型。
若沿用原工作流，在节点 1150 的 JSON 配置中合并以下字段，其余配置保留：

```json
{
  "script": "qwen3vl_guarded_run.py",
  "gemma_thinking_guard": true,
  "penalty_last_n": 4096,
  "stop_on_repetition": true,
  "n_ctx": 32768
}
```

要求原配置为 `chat_handler=gemma4`、`raw_mode=false`，保留 `enable_thinking=true`、`max_tokens=8192`、`repeat_penalty=1.08`、`temperature=1.0`、`top_p=0.95`、`top_k=64`、`min_p=0`。正常驻留沿用节点的 `keep_vram`。
上下文增加可能增加显存占用，实际峰值和耗时需在目标 GPU 验证。

运行时终端应出现：

```text
[Gemma guard 1.0.0] active; penalty_last_n=4096; stop_on_repetition=True
```

## 行为

- 在实际 completion 调用处补传重复惩罚范围，保留处理器原有停止符。
- 流式检测思考区连续四遍的长文本或行组重复，提前失败；不检查最终正文的重复。
- 保留 `finish_reason`；截断、缺失正常结束、未闭合思考和空答案均不提交给后续生图。
- 正常成功保留模型；异常或取消时尝试释放当前缓存，下一次重新加载。
- 错误附带原始输出末尾，不把思考或半截 JSON 当成有效答案；不自动重试或伪造检查通过。
- 现有文本出口可展示错误诊断；此适配不提供 H3 节点的逐 token 网页面板。

本补丁只识别明确的文本重复，不能检测所有语义循环。`max_tokens` 是思考和正文共享的预算。JSON 解析和输出格式校验仍由原工作流负责。

## 兼容范围

已审计 [KLL535/ComfyUI_Simple_Qwen3-VL-gguf](https://github.com/KLL535/ComfyUI_Simple_Qwen3-VL-gguf) 提交 `3e2819d9e1da7b420219ec415f25c98634700969` 的两个文件：

| 文件 | SHA256 |
|---|---|
| `qwen3vl_run.py` | `6b34293f01f634e380ac76a0bd1b4376fa1b205f0b058ed6ea2d8170070f8c27` |
| `qwen3vl_node.py` | `bd2654a20c9b391bd9b2efdd28c949fd3cb4df518b6ab29da6f05b9be2d0a3b9` |

安装前校验原文件指纹；不匹配时停止并显示实际指纹，不覆盖其他版本或本地修改。后端的 `create_completion` 必须明确支持 `penalty_last_n` 及流式调用，否则报 `BACKEND_INCOMPATIBLE`。本包不安装或升级 llama-cpp-python。

备份位于 KLL 插件的 `.gemma-thinking-guard-v1-backup`。已安装文件被其他更新修改后，验证或回滚会拒绝覆盖；此时需要先按实际版本适配。不要强制重置插件。

## 校验与回滚

在本仓库目录执行：

```bash
python3 compat/qwen-gemma-thinking-guard/install.py --plugin /comfyui/custom_nodes/ComfyUI_Simple_Qwen3-VL-gguf --action verify
```

该命令只校验磁盘文件，不能证明当前进程已加载补丁或 GPU 推理兼容。

回滚前停止 ComfyUI，再执行：

```bash
python3 compat/qwen-gemma-thinking-guard/install.py --plugin /comfyui/custom_nodes/ComfyUI_Simple_Qwen3-VL-gguf --action rollback
```

随后重启并使用原 v3 工作流，或把配置的 `script` 改回 `qwen3vl_run.py`、删除 `gemma_thinking_guard`。原始备份保留。

## 测试

29 项离线测试已通过，覆盖原上游函数的 AST 隔离执行、参数透传、重复与截断、取消、缓存释放、安装幂等、指纹拒绝和精确回滚。尚未执行真实 GPU 推理。

安装后可使用原始备份重跑完整测试（测试仅写临时目录）：

```bash
QWEN_GUARD_TEST_SOURCE=/comfyui/custom_nodes/ComfyUI_Simple_Qwen3-VL-gguf/.gemma-thinking-guard-v1-backup python3 -m unittest discover -s compat/qwen-gemma-thinking-guard/tests -v
```

没有上游原文件时，7 项上游/安装测试会跳过；跳过不等于通过。上线前需在实际服务器验证扩写、检查、单次修复及正常驻留。

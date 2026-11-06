# 原生 Gemma 图像扩写与大结构检查

节点类型：`SXDGemmaImageTask`。界面名称：**SXD Gemma · 图像扩写 / 大结构检查**。
两个任务使用同一节点类；在工作流中各放一个实例，`task` 分别为 `prompt`、`anatomy`。

## 更新

先停止 ComfyUI，再执行：

```bash
git -C /comfyui/custom_nodes/ComfyUI-SXD-H3-Prompt pull --ff-only origin main
```

重启 ComfyUI、刷新浏览器，导入 v6 原生节点工作流即可。**没有额外安装脚本，不修改第三方插件。**

如果目录来自压缩包而非 Git 克隆，Git 可能找到外层 ComfyUI 仓库。只有 `git remote get-url origin` 指向 `wskbdwn-dotcom/ComfyUI-SXD-H3-Prompt` 时才运行上述更新命令；不要更改 ComfyUI 主程序的远程地址。

之前发布的 `compat/qwen-gemma-thinking-guard` 已从仓库移除。v4/v5 工作流使用 KLL 节点，需替换为 v6；不能只换配置中的脚本名。此更新不扫描、还原或删除服务器上的其他插件。曾成功应用旧 KLL 补丁的机器，其 KLL 文件不会被本次更新自动恢复；新节点不再加载这些文件。

继续使用与 H3 相同的 Gemma4 GGUF / mmproj 和支持 Gemma4 的 llama-cpp-python 后端。插件更新不升级 CUDA、PyTorch 或推理库。

## 参数与输出

| 输入 | 含义 |
|---|---|
| `config_override` | JSON 模型配置，来自原节点 1150 |
| `system_prompt_override` | 工作流自己的扩写/检查规则，不套用 H3 规则 |
| `user_prompt` | 用户原始要求或检查上下文 |
| `seed` | 推理种子，沿用现有后端的 32 位种子转换 |
| `task` | `prompt` 扩写或 `anatomy` 大结构检查 |
| `release_after_run` | 默认 false，成功后保持模型驻留；true 成功后释放 |
| `image`、`image2`、`image3`、`image4` | 四个图片入口按顺序读取，允许图片批次但合计最多 4 张；检查合计必须恰好一张成图 |

输出 0 `text`：剥离思考、校验并规范键顺序后的 JSON，接原有解析/检查节点。
输出 1 `raw_output`：原始思考与正文；不要接生图提示词。直接连接已有的 `H3PromptLiveDisplay`（H3 · 实时输出查看）即可在执行时查看流式输出；扩写、检查可各接一个面板。v9 工作流已连接这两个面板，v6–v8 可手动连接。

更新插件后需重启 ComfyUI 并刷新浏览器，以加载新增的图像节点事件路由。界面约每 150 毫秒刷新一次，不等待整个推理结束；报错或取消时保留已收到的片段。后台没有浏览器连接时仍正常推理，成功后 `raw_output` 返回全文。预览仅保留最后 200000 字符；内容取决于后端实际暴露的输出，不凭空补写思考。实时查看节点必须直连 `raw_output`，不经过 Set/Get 或其他中间节点。

扩写结构为 `{"prompt":"非空提示词"}`。
检查结构为 `{"status":"pass|uncertain|repair","issues":[]}`。
`repair` 必须有问题，问题字段为 `type/location/evidence/fix`；`pass` 和 `uncertain` 必须是空数组。
大结构类型仅支持 `extra_arm`、`extra_leg`、`extra_head`、`extra_torso`、`missing_arm`、`missing_leg`、`body_twist`、`bad_connection`。
不从分数推导通过状态，不把解析错误、调用失败或截断伪装成 `uncertain`。

## 模型配置

```json
{
  "model_path": "gemma-4-26b-a4b-it-heretic.q4_k_m.gguf",
  "mmproj_path": "gemma-4-26B-A4B-it-heretic-mmproj.bf16.gguf",
  "enable_thinking": true,
  "max_tokens": 8192,
  "n_ctx": 32768,
  "n_gpu_layers": -1,
  "temperature": 1.0,
  "top_k": 64,
  "top_p": 0.95,
  "min_p": 0.0,
  "repeat_penalty": 1.08,
  "penalty_last_n": 4096,
  "stop_on_repetition": true,
  "max_images": 4,
  "image_max_side": 1024,
  "image_jpeg_quality": 90
}
```

支持 `models/LLM`、`models/llm` 下的相对文件名，也支持服务器上实际存在的绝对 GGUF 路径。v6 保留原工作流的绝对模型路径。相对路径出现多个不同匹配文件时会报错。
原 KLL 的 `script`、`chat_handler`、`raw_mode`、`cache_mode` 等参数不属于这个节点；v6 已删除这些字段，传入未知字段会提示使用新版配置。

扩写和检查都保持原有用户规则；模型调用不执行 H3 的六段式后处理。图像按配置的长边限制缩小，不放大；只影响 Gemma 看到的参考图，不更改下游生图的图像输入、分辨率或条件。

## 缓存和异常

H3 与图像节点共享已有模型缓存、锁、加载器和清理逻辑。主模型、投影器、思考开关、上下文和 GPU 层数一致时复用；配置改变则先释放旧模型再加载。不会额外常驻第二套权重。
修改温度、重复惩罚、提示词或图片不重载模型。每轮清理推理状态；开启卸载开关会释放共享的 Gemma，下一次 H3/图像任务会重新加载。原手动释放节点同样作用于这份共享缓存。

重复检查范围在真实 completion 调用处传入；不改变原后端停止符。连续重复的思考、未闭合思考、空答案、非正常结束和超预算截断会中止，错误不提交下游。重复检测不覆盖所有语义循环；8192 是思考和正文的共享生成预算，不是独立思考预算。

## 验证边界

运行 `python3 -m unittest discover -s tests -v` 和 `node tests/test_live.mjs`。
测试覆盖旧 H3 接口、原生节点直接推理、共享驻留、异常释放、规则隔离、多图编码、路径解析、JSON 格式及思考流处理。使用模拟后端，不加载模型。
仍需在目标 GPU 验证实际扩写/检测准确率、显存和耗时；离线测试不等于生产验收。

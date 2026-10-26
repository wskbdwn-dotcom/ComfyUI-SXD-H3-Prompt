# RunPod：T8 与 llama.cpp 库路径

适用已确认模板：ComfyUI 位于 `/comfyui`，Python 位于 `/opt/venv/bin/python3`，启动脚本为 `/comfyui/start.sh`。

已观察到的问题是 Python 包加载到 T8 的旧 `runtime/libllama.so.0`，缺少 `llama_params_fit`。解决方向是让服务优先使用 Python 包配套库，保留原有 CUDA 搜索路径。

## 正在运行的旧 Pod

确认无任务运行后，在终端执行：

```bash
/opt/venv/bin/python3 /comfyui/custom_nodes/ComfyUI-SXD-H3-Prompt/scripts/repair_runpod_library_paths.py
```

此脚本只适用于已有一个运行中的 ComfyUI 服务的上述模板。它读取该服务的环境，在独立子进程中检查后端导入及实际库路径，然后检查、备份并修改 `start.sh`。它不加载模型、不修改 T8、不安装依赖、不自动重启。

看到 `STARTUP` 成功结果后，完整 Restart 当前 Pod，让启动脚本重新执行。仅刷新网页或只在终端 export 不会改变已运行服务的环境。

出现 `STOP` 时保留输出排查；不要盲目移动 T8 或覆盖依赖。脚本备份在 `/comfyui/user/h3_library_path_backups/`，回退时恢复对应 `start.sh` 备份并重启 Pod。

## 公共镜像构建

修复脚本需要运行中的服务，不适合直接在 Docker 构建阶段执行。镜像维护者应把下面配置写入模板 `start.sh`，位于所有环境初始化之后、`export COMFYUI_CMD=` 之前：

```bash
# H3 Gemma4 library paths
export LLAMA_CPP_LIB_PATH=/opt/venv/lib/python3.12/site-packages/llama_cpp/lib
export LD_LIBRARY_PATH="${LLAMA_CPP_LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

已有相同配置时不要重复添加。路径来自已确认的 Python 3.12 模板，换环境后须重新确认。保留原 Python 启动命令、CUDA 路径、T8 文件及其系统配置。

构建前确认现有后端支持 Gemma4ChatHandler 和 enable_thinking；不要把本修复当成任意后端版本兼容保证。确认启动更新、custom_nodes_fix 或挂载目录不会覆盖节点和 start.sh。

使用新版镜像启动测试 Pod，核对实际 ComfyUI 服务的环境及加载的库；再测试两轮扩写、H3 生成和仍需使用的 T8 功能。服务内其他节点也可能使用动态库，单独导入成功不能证明所有插件共存正常。

通过后再把公共模板指向新镜像版本，保留旧版本用于回退。镜像修复固化后，新 Pod 无需重复执行修复脚本。

## 与思考开关无关

上述环境变量解决库选择问题。`enable_thinking` 是工作流中的节点参数，默认 false，直接传给 Gemma4 处理器；不通过环境变量控制，也无需修改第三方源码。

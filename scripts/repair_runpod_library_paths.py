"""Repair the known RunPod start.sh library priority, after an isolated import check."""
from pathlib import Path
from datetime import datetime
import json
import os
import shlex
import shutil
import subprocess
import tempfile


def patched_startup(source, library):
    anchor = 'export COMFYUI_CMD='
    marker = '# H3 Gemma4 library paths'
    block = (marker + '\nexport LLAMA_CPP_LIB_PATH=' + shlex.quote(library) + '\n'
             + 'export LD_LIBRARY_PATH="${LLAMA_CPP_LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n')
    if source.count(anchor) != 1:
        raise ValueError('启动脚本不是已确认的模板，未修改。')
    if marker in source:
        if source.count(marker) != 1 or block + anchor not in source:
            raise ValueError('已有 H3 库路径配置与预期不同，未覆盖。请提供该配置附近的几行。')
        return source
    if 'LLAMA_CPP_LIB_PATH' in source:
        raise ValueError('启动脚本中存在其他 LLAMA_CPP_LIB_PATH 设置，未覆盖。')
    # Only a standalone assignment in the known launch template is supported.
    if not any(line.startswith(anchor) for line in source.splitlines()):
        raise ValueError('启动命令位置不符合已确认模板，未修改。')
    return source.replace(anchor, block + anchor, 1)


def active_comfy_processes(base):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath('cmdline').read_bytes().split(b'\0')
            if str(base / 'main.py').encode() in argv:
                found.append(entry)
        except (OSError, PermissionError):
            continue
    return found


PROBE = r'''
from pathlib import Path
import inspect, json, os
import llama_cpp
try:
    from llama_cpp.llama_multimodal import Gemma4ChatHandler
except ImportError:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
assert 'enable_thinking' in inspect.signature(Gemma4ChatHandler.__init__).parameters, 'missing thinking toggle'
assert callable(getattr(llama_cpp.Llama, 'reset', None)), 'missing Llama.reset'
root = Path(os.environ['LLAMA_CPP_LIB_PATH']).resolve()
loaded = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libllama.so' in line})
assert loaded and all(Path(p).resolve().parent == root for p in loaded), 'unexpected libllama: ' + repr(loaded)
print(json.dumps({'package': getattr(llama_cpp, '__version__', 'unknown'), 'loaded': loaded}, ensure_ascii=False))
'''


def main():
    base = Path('/comfyui')
    python = Path('/opt/venv/bin/python3')
    startup = base / 'start.sh'
    if not python.is_file() or not startup.is_file() or not (base / 'main.py').is_file():
        raise SystemExit('STOP: 未找到已确认的 /comfyui 与 /opt/venv，未修改文件。')
    processes = active_comfy_processes(base)
    if len(processes) != 1:
        raise SystemExit('STOP: 需要恰好一个运行中的 ComfyUI 进程，以复用现有 CUDA 库环境；未修改文件。')
    process = processes[0]
    # Read only to pass the existing environment into a local child. Never print it.
    env = dict(os.environ)
    env.update({os.fsdecode(k): os.fsdecode(v) for item in process.joinpath('environ').read_bytes().split(b'\0')
                if b'=' in item for k,v in [item.split(b'=',1)]})
    print('ComfyUI PID:', process.name)
    print('当前 LLAMA_CPP_LIB_PATH:', env.get('LLAMA_CPP_LIB_PATH', '(未设置)'))
    print('当前 LD_LIBRARY_PATH 首项:', env.get('LD_LIBRARY_PATH','').split(':')[0] or '(未设置)')
    locate = ('import importlib.util; from pathlib import Path; '
              's=importlib.util.find_spec("llama_cpp"); '
              'assert s and s.origin, "llama_cpp not found"; '
              'print(Path(s.origin).resolve().parent / "lib")')
    library = subprocess.check_output([str(python), '-c', locate], env=env, text=True, timeout=30).strip()
    if not Path(library).is_dir():
        raise SystemExit('STOP: 包内动态库目录不存在，未修改文件。')
    original = startup.read_text()
    updated = patched_startup(original, library)
    subprocess.run(['bash', '-n'], input=updated, text=True, check=True)
    # LD_LIBRARY_PATH must be set before the fresh interpreter starts.
    env['LLAMA_CPP_LIB_PATH'] = library
    env['LD_LIBRARY_PATH'] = library + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    probe = subprocess.run([str(python), '-c', PROBE], env=env, text=True, capture_output=True, timeout=90)
    if probe.returncode:
        print(probe.stdout[-2000:])
        print(probe.stderr[-5000:])
        raise SystemExit('STOP: 独立进程验证失败，未修改任何文件。请贴出本次输出。')
    print('IMPORT CHECK:', probe.stdout.strip())
    if updated == original:
        print('STARTUP: 正确的路径设置已存在，没有重复修改。当前运行进程仍需完整重启。')
    else:
        if startup.read_text() != original:
            raise SystemExit('STOP: 检查期间启动脚本有变化，未覆盖。')
        backup_dir = base / 'user/h3_library_path_backups'
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / ('start-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.sh')
        shutil.copy2(startup, backup)
        fd, name = tempfile.mkstemp(prefix='.start-h3-', dir=str(startup.parent))
        try:
            with os.fdopen(fd, 'w') as output:
                output.write(updated)
                output.flush()
                os.fsync(output.fileno())
            shutil.copymode(startup, name)
            os.replace(name, startup)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        print('STARTUP: 已设置优先使用 Python 包自带的 llama.cpp 库。')
        print('BACKUP:', backup)
    print('未修改节点、工作流、T8、依赖或系统 ld.so 配置；没有加载模型。')
    print('NEXT: 任务结束后，在 RunPod 控制台对当前 Pod 执行 Restart，让 start.sh 重新运行。')
    print('这与 Gemma 常驻/释放开关是两回事；只在终端 export 或仅刷新页面不能修复已经加载了错误库的服务。')
    print('如果完整 Restart 后仍出现同一错误，请提供本命令输出和最新报错；不要移动 T8 或重复安装依赖。')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit('STOP: ' + str(error))

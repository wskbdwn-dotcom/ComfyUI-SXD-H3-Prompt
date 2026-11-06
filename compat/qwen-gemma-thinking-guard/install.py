#!/usr/bin/env python3
"""Install/verify/rollback two bounded hooks; never download or upgrade dependencies."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
HELPER = 'qwen3vl_guarded_run.py'
BACKUP = '.gemma-thinking-guard-v1-backup'
HASHES = {
    'qwen3vl_run.py': '6b34293f01f634e380ac76a0bd1b4376fa1b205f0b058ed6ea2d8170070f8c27',
    'qwen3vl_node.py': 'bd2654a20c9b391bd9b2efdd28c949fd3cb4df518b6ab29da6f05b9be2d0a3b9',
}
DECL = 'def _infer_chat(llm, config, images = [], audios = [], videos = []):'
HOOK = '''def _infer_chat(llm, config, images = [], audios = [], videos = []):
    if config.get("gemma_thinking_guard", False):
        from qwen3vl_guarded_run import run_guarded
        return run_guarded(
            _infer_chat_without_guard, llm, config, images, audios, videos,
            release=lambda: unload_llama_model(
                config.get("force_gc_unload", False), config.get("debug", False),
                target=config.get("cache_mode", "keep_vram")))
    return _infer_chat_without_guard(llm, config, images, audios, videos)


def _infer_chat_without_guard(llm, config, images = [], audios = [], videos = []):'''
OLD_DISPATCH = '''            if script_name == "qwen3vl_run.py":
                module = qwen3vl_run'''
NEW_DISPATCH = '''            if script_name == "qwen3vl_guarded_run.py":
                if not config.get("gemma_thinking_guard") or not hasattr(qwen3vl_run, "_infer_chat_without_guard"):
                    return "[ERROR] Gemma guard installation/config incomplete; restart ComfyUI.", None, None
                module = qwen3vl_run
            elif script_name == "qwen3vl_run.py":
                module = qwen3vl_run'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform(name, raw):
    if sha(raw) != HASHES[name]:
        raise RuntimeError(f'版本不匹配：{name} SHA256={sha(raw)}。未覆盖；请提供此指纹以适配实际版本。')
    text = raw.decode('utf-8')
    before, after = (DECL, HOOK) if name == 'qwen3vl_run.py' else (OLD_DISPATCH, NEW_DISPATCH)
    if text.count(before) != 1:
        raise RuntimeError(f'无法唯一定位补丁位置：{name}')
    patched = text.replace(before, after, 1)
    ast.parse(patched, filename=name)
    return patched.encode('utf-8')


def atomic(path, data):
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp, mode)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def install(plugin, action='install'):
    plugin = plugin.resolve()
    helper = (HERE / HELPER).read_bytes()
    ast.parse(helper)
    backup = plugin / BACKUP
    manifest_path = backup / 'manifest.json'
    if manifest_path.exists():
        receipt = json.loads(manifest_path.read_text())
        expected = receipt['installed']
        if set(expected) != {*HASHES, HELPER}:
            raise RuntimeError('备份清单异常，未修改。')
        for name, digest in expected.items():
            file = plugin / name
            if not file.is_file() or sha(file.read_bytes()) != digest:
                raise RuntimeError(f'安装后文件被改动或缺失：{name}；拒绝覆盖，请先核对。')
        if action == 'rollback':
            originals = {name: (backup / name).read_bytes() for name in HASHES}
            for name, data in originals.items():
                if sha(data) != HASHES[name]:
                    raise RuntimeError('原始备份校验失败。')
            for name, data in originals.items():
                atomic(plugin / name, data)
            (plugin / HELPER).unlink()
            # Keep original backup files, retire only the active installation receipt.
            os.replace(manifest_path, backup / 'rolled-back.json')
            return '已回滚。请重启 ComfyUI 并使用 v3 工作流。'
        if sha(helper) != expected[HELPER]:
            raise RuntimeError('补丁包与已安装版本不一致，未覆盖。')
        return '保护补丁文件校验通过。已安装；仍需重启后的 GPU 验证。'
    if action != 'install':
        raise RuntimeError('未检测到完整安装记录。')
    if (plugin / HELPER).exists():
        raise RuntimeError('同名辅助文件已存在，未覆盖。')
    originals = {name: (plugin / name).read_bytes() for name in HASHES}
    patched = {name: transform(name, data) for name, data in originals.items()}
    patched[HELPER] = helper
    # All fingerprints/syntax validated before writing anything to the plugin.
    backup.mkdir(exist_ok=True)
    for name, data in originals.items():
        dest = backup / name
        if dest.exists() and dest.read_bytes() != data:
            raise RuntimeError('已有备份内容不一致，未覆盖。')
    for name, data in originals.items():
        atomic(backup / name, data)
    try:
        atomic(plugin / HELPER, helper)
        for name in HASHES:
            atomic(plugin / name, patched[name])
        atomic(manifest_path, (json.dumps({'version': '1.0.0', 'installed': {
            name: sha(data) for name, data in patched.items()}}, indent=2) + '\n').encode())
    except BaseException:
        for name, data in originals.items():
            atomic(plugin / name, data)
        (plugin / HELPER).unlink(missing_ok=True)
        raise
    return '安装完成。请重启 ComfyUI，然后导入包中的 v4 工作流。'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin', type=Path, required=True)
    parser.add_argument('--action', choices=['install', 'verify', 'rollback'], default='install')
    args = parser.parse_args()
    try:
        print(install(args.plugin, args.action))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + '\n')

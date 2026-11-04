// SPDX-License-Identifier: GPL-3.0-or-later
import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

const LIMIT = 200000;
const views = new Map();
const labels = {starting: '准备 / 加载模型', generating: '正在生成',
    done: '完成', error_or_cancelled: '报错或取消：已保留部分输出'};

function sourceId(node) {
    const input = node.inputs?.find(i => i.name === 'raw_output');
    const link = node.graph?.links?.[input?.link];
    if (!link || link.origin_slot !== 3) return null;
    const source = node.graph.getNodeById(link.origin_id);
    return source?.comfyClass === 'H3AuthorGemma4ExpandV2' ? String(source.id) : null;
}

function render(view, text) {
    const box = view.box;
    const bottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 40;
    view.trimmed ||= text.length > LIMIT;
    view.text = text.slice(-LIMIT);
    box.value = (view.trimmed ? '【预览仅保留末尾 200000 字符；成功后 raw_output 输出包含全文】\n' : '') + view.text;
    if (bottom) box.scrollTop = box.scrollHeight;
}

app.registerExtension({
    name: 'SXD.H3.LiveOutput',
    setup() {
        api.addEventListener('sxd-h3-stream', ({detail: data}) => {
            for (const [node, view] of views) {
                if (sourceId(node) !== String(data.node)) continue;
                if (data.status === 'starting') {
                    view.run = data.run;
                    view.trimmed = false;
                    render(view, '');
                } else if (view.run !== data.run) {
                    continue;
                }
                render(view, view.text + (data.delta || ''));
                const end = data.finish_reason ? ` · finish_reason=${data.finish_reason}` : '';
                const thought = data.thought_end_seen ? ' · 已看到思考结束标记' : '';
                view.status.textContent = `${labels[data.status] || data.status}${thought}${end}`;
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === 'H3AuthorGemma4ExpandV2') {
            const configured = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = configured?.apply(this, arguments);
                if (this.outputs?.length === 3) this.addOutput('raw_output', 'STRING');
                return result;
            };
            return;
        }
        if (nodeData.name !== 'H3PromptLiveDisplay') return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const container = document.createElement('div');
            container.style.cssText = 'display:flex;flex-direction:column;height:100%;gap:6px;background:#202124;color:#eee;padding:8px;box-sizing:border-box;';
            const status = document.createElement('div');
            status.textContent = '将 raw_output 直接接入此节点，运行后实时显示';
            status.style.cssText = 'font:12px sans-serif;white-space:normal;';
            const box = document.createElement('textarea');
            box.readOnly = true;
            box.placeholder = '原始输出：包含思考标记、思考内容及正文。';
            box.style.cssText = 'flex:1;min-height:80px;resize:none;background:#141518;color:#eee;font:13px monospace;white-space:pre-wrap;';
            container.append(status, box);
            this.addDOMWidget('sxd_live_preview', 'custom', container, {serialize: false});
            views.set(this, {box, status, text: '', run: null, trimmed: false});
            this.setSize([520, 360]);
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            const view = views.get(this);
            if (view && Array.isArray(message?.text)) {
                view.trimmed = false;
                render(view, message.text.join('\n'));
                view.status.textContent = '完整输出已就绪（可能来自缓存）';
            }
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            views.delete(this);
            return removed?.apply(this, arguments);
        };
    },
});

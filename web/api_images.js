// SPDX-License-Identifier: GPL-3.0-or-later
import { app } from '../../scripts/app.js';

const imageSlot = /^image_([1-9][0-9]*)$/;

function syncImageInputs(node) {
    if (node._sxdSyncingImages) return;
    node._sxdSyncingImages = true;
    try {
        let lastConnected = 0;
        for (const input of node.inputs || []) {
            const match = imageSlot.exec(input.name);
            if (match && input.link != null) {
                lastConnected = Math.max(lastConnected, Number(match[1]));
            }
        }
        const spareName = `image_${lastConnected + 1}`;
        // Only trim unused trailing sockets. Never rename or move connected inputs.
        for (let i = (node.inputs || []).length - 1; i >= 0; i--) {
            const input = node.inputs[i];
            const match = imageSlot.exec(input.name);
            if (match && input.link == null && Number(match[1]) > lastConnected + 1) {
                node.removeInput(i);
            }
        }
        if (!(node.inputs || []).some(input => input.name === spareName)) {
            node.addInput(spareName, 'IMAGE');
        }
        node.setDirtyCanvas?.(true, true);
    } finally {
        node._sxdSyncingImages = false;
    }
}

app.registerExtension({
    name: 'SXD.APIImageInputs',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'SXDOpenAITextFallback') return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            syncImageInputs(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            // Saved links may still be restoring; reconcile after configuration finishes.
            queueMicrotask(() => syncImageInputs(this));
            return result;
        };
        const changed = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (side, slot, connected) {
            const result = changed?.apply(this, arguments);
            if (side === 1 && imageSlot.test(this.inputs?.[slot]?.name || '')) {
                if (connected) {
                    // Grow synchronously so a following connection can use the new input.
                    // Do not trim while saved links may still be restoring.
                    const next = Math.max(...this.inputs.filter(i => imageSlot.test(i.name))
                        .map(i => Number(imageSlot.exec(i.name)[1]))) + 1;
                    const last = this.inputs.find(i => i.name === `image_${next - 1}`);
                    if (last?.link != null) {
                        this.addInput(`image_${next}`, 'IMAGE');
                    }
                }
                queueMicrotask(() => syncImageInputs(this));
            }
            return result;
        };
    },
});

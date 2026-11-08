// SPDX-License-Identifier: GPL-3.0-or-later
import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

app.registerExtension({
    name: 'SXD.PoseLibraryUpload',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'SXDPoseRoute') return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const picker = document.createElement('input');
            picker.type = 'file';
            picker.accept = '.zip,application/zip';
            picker.hidden = true;
            this.addWidget('button', '上传姿势 JSON 压缩包', '', () => {
                picker.value = '';
                picker.click();
            });
            picker.addEventListener('change', async () => {
                const file = picker.files && picker.files[0];
                if (!file) return;
                const body = new FormData();
                body.append('file', file, file.name);
                try {
                    const response = await api.fetchApi('/sxd/pose-library', {method: 'POST', body});
                    const payload = await response.json();
                    if (!response.ok) throw new Error(payload.error || '上传失败');
                    window.alert(`已保存 ${payload.count} 条。重新运行工作流后使用新数据。`);
                } catch (error) {
                    window.alert(error && error.message ? error.message : '上传失败');
                }
            });
            return result;
        };
    },
});

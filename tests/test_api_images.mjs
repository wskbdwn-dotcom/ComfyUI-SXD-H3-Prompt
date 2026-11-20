// Offline lifecycle checks for autogrowing image inputs; no API/server required.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

let extension;
const app = { registerExtension(value) { extension = value; } };
const source = readFileSync(new URL('../web/api_images.js', import.meta.url), 'utf8')
    .replace(/^import .*;$/gm, '');
vm.runInNewContext(source, { app, queueMicrotask });

class Node {
    constructor() { this.inputs = [{name: 'original_prompt', link: 100}, {name: 'seed', link: 101}]; }
    addInput(name, type) { this.inputs.push({name, type, link: null}); }
    removeInput(index) {
        assert.equal(this.inputs[index].link, null, 'must not remove a connected input');
        this.inputs.splice(index, 1);
    }
    setDirtyCanvas() {}
    onConfigure() { this.originalConfigureCalled = true; }
    onConnectionsChange() { this.originalConnectionCalled = true; }
    connect(name, link) {
        const slot = this.inputs.findIndex(i => i.name === name);
        assert(slot >= 0, `missing input ${name}`);
        this.inputs[slot].link = link;
        this.onConnectionsChange(1, slot, link != null);
    }
}
await extension.beforeRegisterNodeDef(Node, {name: 'SXDOpenAITextFallback'});
const node = new Node();
node.onNodeCreated();
assert.equal(node.inputs.at(-1).name, 'image_1');
node.connect('image_1', 1);
assert.equal(node.inputs.at(-1).name, 'image_2');
node.connect('image_2', 2);
assert.equal(node.inputs.at(-1).name, 'image_3');
await Promise.resolve();
node.connect('image_1', null);
await Promise.resolve();
assert.equal(node.inputs.find(i => i.name === 'image_2').link, 2);
assert.equal(node.inputs.at(-1).name, 'image_3');
// Filling the spare must grow even when a disconnected hole remains earlier.
node.connect('image_3', 3);
assert.equal(node.inputs.at(-1).name, 'image_4');
await Promise.resolve();
node.connect('image_3', null);
await Promise.resolve();
assert.equal(node.inputs.at(-1).name, 'image_3');
node.connect('image_2', null);
await Promise.resolve();
assert.deepEqual(node.inputs.map(i => i.name), ['original_prompt', 'seed', 'image_1']);
assert.equal(node.inputs[0].link, 100);
assert.equal(node.inputs[1].link, 101);
assert(node.originalConnectionCalled);

// Reload a saved graph whose sockets exist before its links have restored.
const restored = new Node();
restored.inputs.push({name: 'image_1', link: null}, {name: 'image_2', link: null},
    {name: 'image_3', link: null});
restored.onConfigure();
restored.connect('image_1', 10);
restored.connect('image_2', 20);
await Promise.resolve();
assert.deepEqual(restored.inputs.slice(2).map(i => [i.name, i.link]), [
    ['image_1', 10], ['image_2', 20], ['image_3', null]]);
assert(restored.originalConfigureCalled);

// Old text-only workflows regain the new optional socket on configure.
const old = new Node();
old.onConfigure();
await Promise.resolve();
assert.equal(old.inputs.at(-1).name, 'image_1');
class Unrelated {}
await extension.beforeRegisterNodeDef(Unrelated, {name: 'OtherNode'});
assert.equal(Unrelated.prototype.onNodeCreated, undefined);
console.log('API image socket lifecycle checks passed.');

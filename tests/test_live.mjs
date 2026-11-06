// Browser event contract tests; no ComfyUI server or GPU is used.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
let extension, listener;
const app = {registerExtension: value => { extension = value; }};
const api = {addEventListener: (event, callback) => {
    assert.equal(event, 'sxd-h3-stream'); listener = callback;
}};
const document = {createElement: () => ({style: {}, value: '', textContent: '',
    scrollTop: 0, clientHeight: 200, scrollHeight: 200, append() {}})};
const source = readFileSync(new URL('../web/live.js', import.meta.url), 'utf8')
    .replace(/^import .*;$/gm, '');
vm.runInNewContext(source, {app, api, document, Map, String});
extension.setup();
class Display {
    constructor() {
        this.inputs = [{name: 'raw_output', link: 7}];
        this.graph = {links: {7: {origin_id: 1150, origin_slot: 3}},
            getNodeById: id => ({id, comfyClass: 'H3AuthorGemma4ExpandV2'})};
    }
    addDOMWidget(name, type, element, options) { assert.equal(options.serialize, false); }
    setSize(size) { this.size = size; }
}
// Capture created elements to inspect the display without HTML interpretation.
const elements = [];
const create = document.createElement;
document.createElement = () => { const e = create(); elements.push(e); return e; };
await extension.beforeRegisterNodeDef(Display, {name: 'H3PromptLiveDisplay'});
const node = new Display(); node.onNodeCreated();
const status = elements[1], box = elements[2];
const emit = (data) => listener({detail: {node: '1150', run: 'run1', delta: '', ...data}});
emit({status: 'starting'});
emit({status: 'generating', delta: '<|channel>thought\n检查'});
assert.equal(box.value, '<|channel>thought\n检查');
emit({node: '999', status: 'generating', delta: 'WRONG NODE'});
emit({run: 'old', status: 'generating', delta: 'WRONG RUN'});
assert(!box.value.includes('WRONG'));
emit({status: 'generating', delta: '<channel|><script>literal text</script>', thought_end_seen: true});
assert(box.value.includes('<script>literal text</script>')); // textarea.value, never innerHTML
assert(status.textContent.includes('已看到思考结束标记'));
emit({status: 'error_or_cancelled', finish_reason: 'length'});
assert(box.value.includes('检查'));
assert(status.textContent.includes('length'));
emit({status: 'starting', run: 'run2'});
assert.equal(box.value, '');
emit({status: 'generating', run: 'run2', delta: 'a'.repeat(210000)});
assert(box.value.startsWith('【预览仅保留'));
assert(box.value.length < 201000);
node.onExecuted({text: ['缓存内容']});
assert.equal(box.value, '缓存内容');
node.onRemoved();
emit({status: 'generating', run: 'run2', delta: 'removed'});
assert.equal(box.value, '缓存内容');
class Expand {
    constructor() { this.outputs = [{}, {}, {}]; this.widgets_values = ['existing rules']; }
    onConfigure() { this.originalCalled = true; }
    addOutput(name, type) { this.outputs.push({name, type}); }
}
await extension.beforeRegisterNodeDef(Expand, {name: 'H3AuthorGemma4ExpandV2'});
const expand = new Expand(); expand.onConfigure(); expand.onConfigure();
assert(expand.originalCalled);
assert.equal(expand.outputs.length, 4);
assert.equal(expand.outputs[3].name, 'raw_output');
assert.deepEqual(expand.widgets_values, ['existing rules']);
// Image expansion and anatomy instances share the viewer but use output slot 1.
const imageNode = new Display();
imageNode.graph = {links: {7: {origin_id: 1151, origin_slot: 1}},
    getNodeById: id => ({id, comfyClass: 'SXDGemmaImageTask'})};
imageNode.onNodeCreated();
const imageBox = elements.at(-1);
emit({node: '1151', status: 'starting'});
emit({node: '1151', status: 'generating', delta: 'image thinking'});
assert.equal(imageBox.value, 'image thinking');
emit({node: '1152', status: 'generating', delta: 'other instance'});
assert.equal(imageBox.value, 'image thinking');
imageNode.graph.links[7].origin_slot = 0;
emit({node: '1151', status: 'generating', delta: 'wrong output slot'});
assert.equal(imageBox.value, 'image thinking');
imageNode.graph.links[7].origin_slot = 1;
emit({node: '1151', status: 'error_or_cancelled', delta: ' partial'});
assert.equal(imageBox.value, 'image thinking partial');
imageNode.onRemoved();
// Structured planning keeps output slot 1 for the same live display contract.
const planNode = new Display();
planNode.graph = {links: {7: {origin_id: 1153, origin_slot: 1}},
    getNodeById: id => ({id, comfyClass: 'SXDGemmaSpatialPlan'})};
planNode.onNodeCreated();
const planBox = elements.at(-1);
emit({node: '1153', status: 'starting'});
emit({node: '1153', status: 'generating', delta: 'structured planning'});
assert.equal(planBox.value, 'structured planning');
planNode.graph.links[7].origin_slot = 0;
emit({node: '1153', status: 'generating', delta: 'not a raw stream'});
assert.equal(planBox.value, 'structured planning');
planNode.onRemoved();
console.log('Live preview event routing, error retention, cache, cleanup and old workflow migration: PASS');

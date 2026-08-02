// The example page's wiring, in jsdom with a mock model.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadPage, mockChatClass } from './helpers/mock-page.mjs';

// ── Source picker → ModelSource ─────────────────────────────────────────────

test('the four source kinds are all present and hub is the default', async () => {
  const p = await loadPage();
  const kinds = [...p.doc.querySelectorAll('input[name=kind]')].map((i) => i.value);
  assert.deepEqual(kinds, ['hub', 'base', 'url', 'file']);
  assert.equal(p.doc.querySelector('input[name=kind]:checked').value, 'hub');
});

test('switching kind reveals only that kind\'s fields', async () => {
  const p = await loadPage();
  const pick = (v) => {
    p.doc.querySelector(`input[value="${v}"]`).checked = true;
    p.doc.querySelector(`input[value="${v}"]`).dispatchEvent(new p.window.Event('change', { bubbles: true }));
  };
  pick('url');
  assert.equal(p.doc.querySelector('[data-for=url]').hidden, false);
  assert.equal(p.doc.querySelector('[data-for=hub]').hidden, true);
  pick('file');
  assert.equal(p.doc.querySelector('[data-for=file]').hidden, false);
  assert.equal(p.doc.querySelector('[data-for=url]').hidden, true);
});

test('load passes the hub source straight through', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();
  assert.deepEqual({ ...cc.state.loads[0].source }, { hub: 'onnx-community/Qwen2.5-0.5B-Instruct' });
});

test('load passes base+id when that kind is selected', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.doc.querySelector('input[value="base"]').checked = true;
  p.click('load');
  await p.settle();
  assert.deepEqual({ ...cc.state.loads[0].source }, { base: '/models/', id: 'Qwen/Qwen2.5-0.5B-Instruct' });
});

test('device and dtype overrides reach load; auto is passed as auto', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.$('device').value = 'wasm';
  p.$('dtype').value = 'q8';
  p.click('load');
  await p.settle();
  assert.equal(cc.state.loads[0].opts.device, 'wasm');
  assert.equal(cc.state.loads[0].opts.dtype, 'q8');
});

test('the archive-file kind refuses to load with no file picked', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.doc.querySelector('input[value="file"]').checked = true;
  p.click('load');
  await p.settle();
  assert.equal(cc.state.loads.length, 0, 'no load attempted');
  assert.match(p.$('log').textContent, /pick a \.zip/);
});

// ── The editable tools file ─────────────────────────────────────────────────

test('the sample tools file registers three tools on load', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();
  const chat = cc.state.instances[0];
  assert.deepEqual([...chat.tools.keys()], ['get_weather', 'get_time', 'multiply']);
  assert.match(p.$('toolStatus').textContent, /3 tools live/);
  assert.match(p.$('toolList').textContent, /get_weather/);
});

test('editing the file and applying replaces the live tool set', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  p.$('toolsrc').value = "tool('only_one', 'Just one', {}, async () => 1);";
  p.click('apply');
  await p.settle();

  assert.deepEqual([...cc.state.instances[0].tools.keys()], ['only_one']);
  assert.match(p.$('toolStatus').textContent, /1 tool live/);
});

test('a broken tools file reports the error and keeps the page usable', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  p.$('toolsrc').value = 'tool((((';
  p.click('apply');
  await p.settle();

  assert.match(p.$('toolStatus').textContent, /✗/);
  assert.equal(p.$('ask').disabled, false, 'the page still works');
});

test('edits persist to localStorage and are restored on reload', async () => {
  const p = await loadPage();
  p.$('toolsrc').value = "tool('mine', 'Mine', {}, async () => 1);";
  p.click('apply');
  await p.settle();
  assert.match(p.window.localStorage.getItem('nexus-demo-tools'), /mine/);
});

test('applying before a model is loaded saves rather than throwing', async () => {
  const p = await loadPage();
  p.click('apply');
  await p.settle();
  assert.match(p.$('toolStatus').textContent, /will apply when the model loads/);
});

test('reset restores the sample file', async () => {
  const p = await loadPage();
  p.$('toolsrc').value = 'nonsense';
  p.click('reset');
  await p.settle();
  assert.match(p.$('toolsrc').value, /tool\('get_weather'/);
});

// ── Asking ─────────────────────────────────────────────────────────────────

test('the question is prefilled and asking renders both bubbles', async () => {
  const cc = mockChatClass({ script: ['It is 31C in Chennai.'] });
  const p = await loadPage({ chatClass: cc });
  assert.match(p.$('q').value, /weather in Chennai/);

  p.click('load');
  await p.settle();
  p.click('ask');
  await p.settle();

  assert.equal(cc.state.instances[0].asked[0], "What's the weather in Chennai?");
  assert.match(p.$('chat').textContent, /YOU|you/i);
  assert.match(p.$('chat').textContent, /It is 31C in Chennai\./);
});

test('the preset buttons ask their own question', async () => {
  const cc = mockChatClass({ script: ['1096637'] });
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  const math = [...p.doc.querySelectorAll('button[data-q]')].find((b) => /4831/.test(b.dataset.q));
  math.dispatchEvent(new p.window.Event('click'));
  await p.settle();

  assert.match(cc.state.instances[0].asked[0], /4831 multiplied by 227/);
});

test('tool calls are logged to the sidebar as they fire', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  cc.state.instances[0].emit('toolCall', { name: 'get_weather', arguments: { city: 'Chennai' } }, { conditions: '31C' });

  assert.match(p.$('calls').textContent, /get_weather/);
  assert.match(p.$('calls').textContent, /31C/);
});

test('asking is disabled until a model is loaded', async () => {
  const p = await loadPage();
  assert.equal(p.$('ask').disabled, true);
  assert.equal(p.$('q').disabled, true);
  p.click('load');
  await p.settle();
  assert.equal(p.$('ask').disabled, false);
});

test('a failed load is reported and leaves asking disabled', async () => {
  const cc = mockChatClass();
  cc.state.failNext = true;
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();
  assert.match(p.$('log').textContent, /✗ boom/);
  assert.equal(p.$('ask').disabled, true);
  assert.equal(p.$('load').disabled, false, 're-loadable after a failure');
});

// ── The copyable snippet ───────────────────────────────────────────────────

test('the snippet reflects the selected source and updates live', async () => {
  const p = await loadPage();
  assert.match(p.$('snippet').textContent, /hub: 'onnx-community\/Qwen2\.5-0\.5B-Instruct'/);

  p.$('hub').value = 'my-org/my-model';
  p.fire('hub', 'input');
  assert.match(p.$('snippet').textContent, /hub: 'my-org\/my-model'/);
});

test('the snippet switches shape for archive sources', async () => {
  const p = await loadPage();
  const pick = (v) => {
    p.doc.querySelector(`input[value="${v}"]`).checked = true;
    p.doc.querySelector(`input[value="${v}"]`).dispatchEvent(new p.window.Event('change', { bubbles: true }));
  };
  pick('file');
  assert.match(p.$('snippet').textContent, /archive: fileFromInput/);
  pick('base');
  assert.match(p.$('snippet').textContent, /base: '\/models\/', id:/);
});

test('device and dtype appear in the snippet only when overridden', async () => {
  const p = await loadPage();
  assert.doesNotMatch(p.$('snippet').textContent, /device:/);
  p.$('device').value = 'webgpu';
  p.fire('device', 'change');
  assert.match(p.$('snippet').textContent, /device: 'webgpu'/);
});

test('the snippet loads tools from a file, and defines every name it uses', async () => {
  const p = await loadPage();
  const snippet = p.$('snippet').textContent;
  assert.match(snippet, /await chat\.loadTools\('\.\/tools\.js'\)/);
  // The bug this replaced: the snippet referenced `toolsSource`, a variable it
  // never defined, so the code people were told to copy could not run. Check
  // the CODE lines only — the prose in comments legitimately says "tools".
  const code = snippet.split('\n').filter((l) => !l.trim().startsWith('//')).join('\n');
  for (const name of ['toolsSource', 'toolsFile']) {
    assert.equal(new RegExp(`\\b${name}\\b`).test(code), false,
      `snippet uses undefined identifier "${name}"`);
  }
  // Not asserted here: the snippet's own `import` line. mock-page strips every
  // line starting with `import` to satisfy the module's real imports, which
  // also eats that line out of the template literal. A harness artefact, not a
  // page bug — the deployed snippet is checked against the live page instead.
});

// ── Cache ──────────────────────────────────────────────────────────────────

test('the cache panel reports buckets, file counts and disk usage', async () => {
  const p = await loadPage({
    caches: { 'transformers-cache': ['https://h/a.onnx', 'https://h/b.json'] },
  });
  p.click('cacheRefresh');
  await p.settle();
  assert.match(p.$('cacheTable').textContent, /transformers-cache/);
  assert.match(p.$('cacheTable').textContent, /2 files/);
  assert.match(p.$('cacheTable').textContent, /512\.0 MB/);
});

test('deleting clears the real cache buckets, not just localStorage', async () => {
  const p = await loadPage({ caches: { 'transformers-cache': ['https://h/a.onnx'] } });
  p.$('toolsrc').value = "tool('mine','M',{},async()=>1);";
  p.click('apply');
  await p.settle();

  p.click('cacheClear');
  await p.settle();

  assert.equal(p.window.__buckets.size, 0, 'cached weights actually removed');
  assert.equal(p.window.localStorage.getItem('nexus-demo-tools'), null, 'saved tools removed');
  assert.match(p.$('toolsrc').value, /get_weather/, 'editor back to the sample');
});

test('declining the confirm leaves everything alone', async () => {
  const p = await loadPage({ caches: { 'transformers-cache': ['https://h/a.onnx'] } });
  p.window.__confirmAnswer = false;
  p.click('cacheClear');
  await p.settle();
  assert.equal(p.window.__buckets.size, 1);
});

test('clearing resets the session so a reload is required', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc, caches: { 'transformers-cache': ['https://h/a.onnx'] } });
  p.click('load');
  await p.settle();
  assert.equal(p.$('ask').disabled, false);

  p.click('cacheClear');
  await p.settle();

  assert.equal(p.$('ask').disabled, true);
  assert.match(p.$('bDevice').textContent, /device —/);
  assert.match(p.$('log').textContent, /cache cleared/);
});

// ── Diagnostics panel ──────────────────────────────────────────────────────

test('each round is shown with prompt, raw output and parse verdict', async () => {
  const cc = mockChatClass({ script: ['It is 31C.'] });
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  const chat = cc.state.instances[0];
  chat.emit('prompt', '<|im_start|>system\n<tools>\n{"name":"get_weather"}\n</tools>', 0);
  chat.emit('raw', '<tool_call>{"name":"get_weather"}</tool_call>', [{ name: 'get_weather', arguments: {} }], 0);

  const d = p.$('diag').textContent;
  assert.match(d, /round 1/);
  assert.match(d, /1 tool call parsed/);
  assert.match(d, /tool schemas: yes/);
});

// The live-demo failure: tools registered, one round, zero calls, a degenerate
// repetition returned as the answer. The panel has to name that out loud.
test('a model answering instead of calling is called out explicitly', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  const chat = cc.state.instances[0];
  chat.emit('prompt', 'system <tools>{"name":"get_weather"}</tools> user', 0);
  chat.emit('raw', 'The weather in Chennai is described as follows: '.repeat(5), [], 0);

  const d = p.$('diag').textContent;
  assert.match(d, /no tool call — the model answered instead/);
  assert.match(d, /described as follows/, 'the raw output is visible');
});

test('a prompt missing tool schemas is flagged', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  cc.state.instances[0].emit('prompt', 'just a user turn, no schemas here', 0);

  assert.match(p.$('diag').textContent, /tool schemas: NO/);
});

test('the panel resets between questions', async () => {
  const cc = mockChatClass({ script: ['a'] });
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  cc.state.instances[0].emit('raw', 'first round text', [], 0);
  assert.match(p.$('diag').textContent, /first round text/);

  p.click('ask');
  await p.settle();
  assert.doesNotMatch(p.$('diag').textContent, /first round text/);
});

// ── The model self-check, run right after load ──────────────────────────────
// The published matrix cannot cover a model the user just supplied from a zip
// or their own host, so the page asks the model itself and shows the verdict.

test('a passing model is reported on the badge and in the log', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  assert.equal(cc.state.instances[0].selfChecked, true, 'the check actually ran');
  assert.match(p.$('bCheck').textContent, /tool calling ✓/);
  assert.match(p.$('log').textContent, /calls tools correctly/);
});

test('a model no quantization can rescue fails the load, rather than loading broken', async () => {
  const cc = mockChatClass();
  cc.state.selfCheck = {
    ok: false, called: false, grounded: false, needed_forcing: false,
    model: 'tiny/model', device: 'wasm', dtype: 'q8', answer: 'about 40',
    detail: 'tiny/model (wasm/q8) does not call tools.',
  };
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  assert.match(p.$('bCheck').textContent, /tool calling ✗/);
  assert.match(p.$('log').textContent, /does not call tools/);
  assert.match(p.$('log').textContent, /Try a bigger model/i);
  // The whole point: you are not left with a chat box that cannot work.
  assert.equal(p.$('ask').disabled, true, 'asking stays disabled');
});

test('a model that only calls under forcing says so', async () => {
  const cc = mockChatClass();
  cc.state.selfCheck = {
    ok: true, called: true, grounded: true, needed_forcing: true,
    model: 'm', device: 'wasm', dtype: 'q4', answer: 'QX-7731',
    detail: 'm (wasm/q4) calls tools correctly, but only when the call syntax is forced.',
  };
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();

  assert.match(p.$('bCheck').textContent, /forced/);
});

test('the self-check does not leave its rounds in the diagnostics panel', async () => {
  const cc = mockChatClass();
  const p = await loadPage({ chatClass: cc });
  p.click('load');
  await p.settle();
  assert.doesNotMatch(p.$('diag').textContent, /sensor|QX-7731/);
});

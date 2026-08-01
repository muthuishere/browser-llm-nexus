export interface ToolCall {
  name: string;
  arguments: Record<string, unknown> | string;
}

/** Parse tool calls from raw LLM output into structured objects.
 *  Handles Qwen/Hermes <tool_call>, Mistral [TOOL_CALLS], Llama bare JSON,
 *  fenced JSON, nested OpenAI-style function objects, string-encoded args. */
export function parseToolCalls(text: string): ToolCall[] {
  const tryJSON = (s: string): unknown => {
    try { return JSON.parse(s); } catch { return null; }
  };
  const calls: ToolCall[] = [];
  const push = (o: unknown): void => {
    if (!o || typeof o !== 'object') return;
    const obj = o as Record<string, any>;
    const name: unknown = obj.name ?? obj.function?.name;
    let args: unknown = obj.arguments ?? obj.parameters ?? obj.function?.arguments ?? {};
    if (typeof args === 'string') args = tryJSON(args) ?? args;
    if (typeof name === 'string' && name) {
      calls.push({ name, arguments: args as ToolCall['arguments'] });
    }
  };
  for (const m of text.matchAll(/<tool_call>\s*([\s\S]*?)\s*(?:<\/tool_call>|$)/g)) push(tryJSON(m[1]!));
  if (!calls.length) {
    const m = text.match(/\[TOOL_CALLS\]\s*(\[[\s\S]*?\])/);
    if (m) for (const c of (tryJSON(m[1]!) as unknown[]) ?? []) push(c);
  }
  if (!calls.length) {
    const body = (text.match(/```(?:json)?\s*([\s\S]*?)```/)?.[1] ?? text).trim();
    const parsed = tryJSON(body) ?? tryJSON(body.match(/\{[\s\S]*\}/)?.[0] ?? '');
    Array.isArray(parsed) ? parsed.forEach(push) : push(parsed);
  }
  return calls;
}

/** Strip <think>…</think> traces reasoning models prepend. */
export const stripThinking = (t: string): string =>
  t.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim();

/** Salvage a tool call from a *primed* generation.
 *
 *  When the library forces a call it hands the model an already-open
 *  `<tool_call>\n{"name": "` and lets it continue. Small models frequently
 *  produce something that is nearly-but-not-quite JSON: an unterminated
 *  object, a trailing explanation, a stray newline inside the braces. Strict
 *  parsing throws all of that away and the forced turn is wasted — which is
 *  exactly what "FORCE-FAILED" was: the model named the right tool and we
 *  discarded it over a missing brace.
 *
 *  So: try the strict parser first, then fall back to pulling the name and
 *  arguments out with regexes, then match the name against what is actually
 *  registered (exact, then case-insensitive, then unique substring). Returns
 *  null when nothing trustworthy can be recovered. */
export function salvageToolCall(text: string, known: readonly string[]): ToolCall | null {
  const isKnown = (n: string): string | null => {
    if (known.includes(n)) return n;
    const lower = known.filter((k) => k.toLowerCase() === n.toLowerCase());
    if (lower.length === 1) return lower[0]!;
    // A model that writes "weather" for "get_weather" meant the one tool that
    // contains it — but only when exactly one does, or it is a guess.
    const partial = known.filter((k) => k.includes(n) || n.includes(k));
    return partial.length === 1 ? partial[0]! : null;
  };

  for (const c of parseToolCalls(text)) {
    const name = isKnown(c.name);
    if (name) return { name, arguments: c.arguments };
  }

  const name = isKnown((text.match(/"name"\s*:\s*"([^"]+)"/) ?? text.match(/^\s*([a-zA-Z_][\w]*)"/))?.[1] ?? '');
  if (!name) return null;

  // Arguments are best-effort: the call is worth running even without them
  // (a zero-arg tool, or one whose defaults are fine) — better than nothing.
  let args: Record<string, unknown> = {};
  const block = text.match(/"(?:arguments|parameters)"\s*:\s*(\{[\s\S]*)/)?.[1];
  if (block) {
    // Walk to the matching brace; the tail after it is usually prose.
    let depth = 0;
    let end = -1;
    for (let i = 0; i < block.length; i++) {
      if (block[i] === '{') depth++;
      else if (block[i] === '}' && --depth === 0) { end = i + 1; break; }
    }
    const candidate = end > 0 ? block.slice(0, end) : block + '}'.repeat(Math.max(depth, 1));
    try { args = JSON.parse(candidate) as Record<string, unknown>; } catch { /* keep {} */ }
  }
  return { name, arguments: args };
}

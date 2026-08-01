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

// The spawn probe, ported to the TypeScript agent frameworks.
//
//   node spawn.mjs <vercel|mastra> <depth> <limit> <baseURL>
//
// Same tree as broods/probes/spawn: one `delegate` tool per agent; at depth 0
// it returns a string, deeper it builds a fresh agent the same way and runs it
// on the subtask. Every agent declares the same limit through its framework's
// documented step bound. The Python driver (broods/experiments/ts_spawn.py)
// arms the proxy and counts calls at the socket; this script only reports how
// many agents it built and how the root run ended.
import { generateText, tool, stepCountIs } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { z } from 'zod';

const [fw, depthS, limitS, baseURL] = process.argv.slice(2);
const DEPTH = Number(depthS), L = Number(limitS);
const SYSTEM_PROMPT = 'You are a research agent. Break the task into subtasks and ' +
  'delegate each one using the delegate tool before answering.';
const provider = createOpenAICompatible({ name: 'broods', baseURL, apiKey: 'unused',
                                          includeUsage: true });
const model = () => provider.chatModel('mock');
let agents = 0;

async function child(d, task) {
  if (d === 0) return 'leaf complete';
  try { await run(d - 1, task); return 'child complete'; }
  catch (e) { return `child stopped: ${e?.name ?? 'Error'}`; }   // a conscientious tool
}

async function runVercel(d, task) {
  agents++;
  const delegate = tool({
    description: 'Delegate a subtask to a helper agent.',
    inputSchema: z.object({ task: z.string() }),
    execute: async ({ task }) => child(d, task),
  });
  // stopWhen: stepCountIs(L) is the AI SDK's documented step bound.
  await generateText({ model: model(), system: SYSTEM_PROMPT, prompt: task,
                       tools: { delegate }, stopWhen: stepCountIs(L) });
}

async function runMastra(d, task) {
  agents++;
  const { Agent } = await import('@mastra/core/agent');
  const { createTool } = await import('@mastra/core/tools');
  const delegate = createTool({
    id: 'delegate',
    description: 'Delegate a subtask to a helper agent.',
    inputSchema: z.object({ task: z.string() }),
    // Mastra has passed the input as `context`, as the first argument, and as
    // `inputData` across versions; accept each rather than pin one.
    execute: async (a) => child(d, a?.context?.task ?? a?.inputData?.task ?? a?.task ?? ''),
  });
  const agent = new Agent({ name: `agent_${agents}`, instructions: SYSTEM_PROMPT,
                            model: model(), tools: { delegate } });
  // maxSteps is Mastra's documented per-generate step bound.
  await agent.generate(task, { maxSteps: L });
}

const run = fw === 'mastra' ? runMastra : runVercel;
let outcome = 'COMPLETED', error = null;
try { await run(DEPTH, 'Research a topic.'); }
catch (e) { outcome = e?.name ?? 'Error'; error = String(e?.message ?? e).slice(0, 200); }
const pkg = fw === 'mastra' ? '@mastra/core' : 'ai';
const { createRequire } = await import('node:module');
const version = createRequire(import.meta.url)(`${pkg}/package.json`).version;
console.log(JSON.stringify({ framework: fw, version, depth: DEPTH, agents_built: agents,
                             outcome, error }));

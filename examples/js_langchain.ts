/**
 * LangChain.js + Distil proxy example.
 *
 * Run `distil proxy` first, then point the SDK at it:
 *
 *   distil proxy --port 8788 --upstream https://api.anthropic.com
 *
 * Install:
 *   npm install @langchain/anthropic
 *
 * Run with:
 *   ANTHROPIC_API_KEY=sk-ant-… npx tsx examples/js_langchain.ts
 */

import { ChatAnthropic } from "@langchain/anthropic";

// ChatAnthropic accepts `anthropicApiUrl` to override the base URL.
// All messages traffic goes through the local proxy before reaching
// the real Anthropic API.
//
// Note: if you are on an older version of @langchain/anthropic that does
// not yet expose `anthropicApiUrl`, use the `clientOptions.baseURL` field
// instead:
//   clientOptions: { baseURL: "http://127.0.0.1:8788" }
const model = new ChatAnthropic({
  model: "claude-opus-4-5",
  apiKey: process.env.ANTHROPIC_API_KEY,
  anthropicApiUrl: "http://127.0.0.1:8788",
});

const response = await model.invoke([
  ["system", "You are a helpful assistant."],
  ["human", "Summarise the key ideas behind prompt compression."],
]);

console.log(response.content);

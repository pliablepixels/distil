/**
 * Vercel AI SDK + Distil proxy example.
 *
 * Run `distil proxy` first, then point the SDK at it:
 *
 *   distil proxy --port 8788 --upstream https://api.anthropic.com
 *
 * Install the AI SDK provider:
 *   npm install @ai-sdk/anthropic ai
 *
 * Run with:
 *   ANTHROPIC_API_KEY=sk-ant-… npx tsx examples/js_vercel_ai_sdk.ts
 */

import { createAnthropic } from "@ai-sdk/anthropic";
import { generateText } from "ai";

// createAnthropic accepts a baseURL that overrides the default upstream.
// The proxy sits at http://127.0.0.1:8788 and forwards /v1/messages to
// the real Anthropic API after compressing the messages array.
const anthropic = createAnthropic({
  baseURL: "http://127.0.0.1:8788",
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const { text } = await generateText({
  model: anthropic("claude-opus-4-5"),
  system: "You are a helpful assistant.",
  prompt: "Summarise the key ideas behind prompt compression.",
});

console.log(text);

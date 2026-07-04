import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import type { Model } from "../src/types.ts";

// The openai-completions provider historically only read
// prompt_tokens_details.cache_write_tokens (an OpenRouter-ism), so cacheWrite was
// always reported as 0 behind a LiteLLM gateway. LiteLLM reports Anthropic
// prompt-caching writes as prompt_tokens_details.cache_creation_tokens (a public
// field on its PromptTokensDetailsWrapper). These tests drive the real stream path.

const mockState = vi.hoisted(() => ({
	usage: undefined as Record<string, unknown> | undefined,
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: () => {
					const usage = mockState.usage;
					const stream = {
						async *[Symbol.asyncIterator]() {
							yield {
								id: "chatcmpl-test",
								choices: [{ delta: {}, finish_reason: "stop" }],
								usage,
							};
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}

	return { default: FakeOpenAI };
});

const model: Model<"openai-completions"> = {
	id: "custom-model",
	name: "Custom Model",
	api: "openai-completions",
	provider: "openrouter",
	baseUrl: "https://example.com/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128000,
	maxTokens: 32000,
};

async function cacheWriteFor(usage: Record<string, unknown>): Promise<number> {
	mockState.usage = usage;
	const result = await streamOpenAICompletions(
		model,
		{
			systemPrompt: "System prompt",
			messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
		},
		{ apiKey: "test-key" },
	).result();
	return result.usage.cacheWrite;
}

describe("openai-completions parseChunkUsage cache-write fallbacks", () => {
	beforeEach(() => {
		mockState.usage = undefined;
	});

	it("reads LiteLLM cache writes nested as prompt_tokens_details.cache_creation_tokens", async () => {
		const cacheWrite = await cacheWriteFor({
			prompt_tokens: 19,
			completion_tokens: 3,
			prompt_tokens_details: { cached_tokens: 0, cache_creation_tokens: 7277 },
		});
		expect(cacheWrite).toBe(7277);
	});

	it("keeps OpenRouter cache_write_tokens as the highest-priority source", async () => {
		const cacheWrite = await cacheWriteFor({
			prompt_tokens: 1000,
			completion_tokens: 3,
			prompt_tokens_details: { cached_tokens: 0, cache_write_tokens: 100, cache_creation_tokens: 999 },
		});
		expect(cacheWrite).toBe(100);
	});

	it("reports 0 when no cache-write field is present", async () => {
		const cacheWrite = await cacheWriteFor({
			prompt_tokens: 10,
			completion_tokens: 3,
			prompt_tokens_details: { cached_tokens: 0 },
		});
		expect(cacheWrite).toBe(0);
	});
});

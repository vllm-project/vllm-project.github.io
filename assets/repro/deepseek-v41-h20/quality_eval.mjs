import { writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";

const revision = "7cf1290ed87c28a31f867e0f47a7cb62a61d502e";
const datasetUrl = "https://datasets-server.huggingface.co/rows";
const baseUrl = process.env.BASELINE_URL ?? "http://127.0.0.1:18080";
const tunedUrl = process.env.TUNED_URL ?? "http://127.0.0.1:18081";
const sampleCount = Number(process.env.SAMPLE_COUNT ?? 500);
const sampleSeed = Number(process.env.SAMPLE_SEED ?? 20260926);
const model = "deepseek-v4-1-flash-tp8";
const prompt =
  "Solve this grade-school math problem. Show your work and put the final " +
  "numeric answer after ####.\n\n";

async function loadDataset() {
  const rows = [];
  for (let offset = 0; offset < 1319; offset += 100) {
    const url = new URL(datasetUrl);
    url.search = new URLSearchParams({
      dataset: "openai/gsm8k",
      config: "main",
      split: "test",
      offset: String(offset),
      length: String(Math.min(100, 1319 - offset)),
      revision,
    });
    const page = execFileSync("curl", ["-fsS", url], { encoding: "utf8" });
    rows.push(...JSON.parse(page).rows.map(({ row }) => row));
  }
  if (rows.length !== 1319) throw new Error(`Expected 1319 rows, got ${rows.length}`);
  return rows;
}

function sampleIndices(length, count, seed) {
  const indices = Array.from({ length }, (_, index) => index);
  let state = seed >>> 0;
  for (let i = indices.length - 1; i > 0; i -= 1) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    const j = (state >>> 0) % (i + 1);
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices.slice(0, count);
}

function answerNumber(text) {
  const boxed = [...text.matchAll(/\\boxed\{\s*([^{}]+)\s*\}/g)].at(-1)?.[1];
  const marked = [...text.matchAll(/####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)/g)].at(-1)?.[1];
  const lastLine = text.trim().split(/\r?\n/).at(-1) ?? "";
  const finalNumber = [...lastLine.matchAll(/[-+]?\$?\d[\d,]*(?:\.\d+)?/g)].at(-1)?.[0];
  const value = boxed ?? marked ?? finalNumber;
  return value?.replace(/[,$\s]/g, "") ?? null;
}

function expectedNumber(answer) {
  return answer.match(/####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)/)?.[1]
    ?.replace(/[,$\s]/g, "") ?? null;
}

async function query(url, question) {
  const started = performance.now();
  const response = await fetch(`${url}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt + question }],
      temperature: 0,
      top_p: 1,
      seed: 42,
      max_tokens: 2048,
    }),
  });
  const elapsedMs = Math.round(performance.now() - started);
  if (!response.ok) throw new Error(`Inference failed: ${response.status}`);
  const result = await response.json();
  return {
    prediction: answerNumber(
      `${result.choices?.[0]?.message?.reasoning ?? ""}\n${result.choices?.[0]?.message?.content ?? ""}`,
    ),
    finish_reason: result.choices?.[0]?.finish_reason ?? null,
    elapsed_ms: elapsedMs,
  };
}

if (!Number.isInteger(sampleCount) || sampleCount < 1 || sampleCount > 1319) {
  throw new Error("SAMPLE_COUNT must be between 1 and 1319");
}

const rows = await loadDataset();
const indices = sampleIndices(rows.length, sampleCount, sampleSeed);
const results = [];
for (const index of indices) {
  const row = rows[index];
  const expected = expectedNumber(row.answer);
  const [baseline, tuned] = await Promise.all([
    query(baseUrl, row.question),
    query(tunedUrl, row.question),
  ]);
  results.push({
    dataset_index: index,
    expected,
    baseline,
    tuned,
    baseline_correct:
      baseline.finish_reason !== "length" && baseline.prediction === expected,
    tuned_correct:
      tuned.finish_reason !== "length" && tuned.prediction === expected,
  });
  if ((results.length % 10) === 0) {
    console.log(`completed ${results.length}/${sampleCount}`);
  }
}

const score = (key) => results.filter((result) => result[key]).length;
const baselineCompleted = results.filter(
  (result) => result.baseline.finish_reason !== "length",
).length;
const tunedCompleted = results.filter(
  (result) => result.tuned.finish_reason !== "length",
).length;
const pairedCompleted = results.filter(
  (result) =>
    result.baseline.finish_reason !== "length" &&
    result.tuned.finish_reason !== "length",
);
const summary = {
  dataset: "openai/gsm8k main/test",
  dataset_revision: revision,
  baseline_recipe: "original MXFP4 checkpoint, Humming W4A8 MoE",
  tuned_recipe: "converted FP8-128 checkpoint, FlashInfer CUTLASS MoE",
  sample_count: sampleCount,
  sample_seed: sampleSeed,
  prompt,
  decoding: { temperature: 0, top_p: 1, seed: 42, max_tokens: 2048 },
  baseline_completed: baselineCompleted,
  tuned_completed: tunedCompleted,
  paired_completed: pairedCompleted.length,
  baseline_correct: score("baseline_correct"),
  tuned_correct: score("tuned_correct"),
  baseline_accuracy: score("baseline_correct") / baselineCompleted,
  tuned_accuracy: score("tuned_correct") / tunedCompleted,
  both_correct: pairedCompleted.filter(
    (r) => r.baseline_correct && r.tuned_correct,
  ).length,
  baseline_only: pairedCompleted.filter(
    (r) => r.baseline_correct && !r.tuned_correct,
  ).length,
  tuned_only: pairedCompleted.filter(
    (r) => !r.baseline_correct && r.tuned_correct,
  ).length,
  neither_correct: pairedCompleted.filter(
    (r) => !r.baseline_correct && !r.tuned_correct,
  ).length,
  baseline_truncated: results.filter((r) => r.baseline.finish_reason === "length").length,
  tuned_truncated: results.filter((r) => r.tuned.finish_reason === "length").length,
};

await writeFile("quality-results.jsonl", results.map((row) => JSON.stringify(row)).join("\n") + "\n");
await writeFile("quality-summary.json", `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));

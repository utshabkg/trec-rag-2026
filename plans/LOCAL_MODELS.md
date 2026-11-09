# Local LLM serving — what works on this machine, and why

Verified empirically on 2026-08-01, not inferred from docs. Reusable across
projects, so it records the *constraint* as well as the recipes.

**Machine:** 2× RTX 6000 Ada (48 GB each) · driver **575.57.08 = CUDA 12.9** ·
shared with other users, who regularly hold ~47 GB via Ollama.

---

## Verified roster

| Model | vLLM | Notes |
|---|---|---|
| **Qwen2.5-72B-Instruct-AWQ** | ✅ works | AWQ 4-bit, 38.7 GB (19.4 GiB/GPU at tp=2). Needs both cards nearly idle. Ran all 119 narratives, 119/119 first attempt. |
| **Qwen2.5-7B-Instruct** | ✅ works | bf16, 15.2 GB. Fits on **one** GPU (`--tensor-parallel-size 1`), so it runs alongside another user's job. Verified 8/8. |
| **Gemma 4 31B IT** | ❌ blocked | Not a model problem — a **driver ceiling**. See below. Use Ollama instead. |

### Launch recipes

```bash
conda activate trec-rag-2026-vllm

# Qwen 72B AWQ — highest quality; needs a free-GPU window
python scripts/decompose/preprocess_step1_decompose.py \
  --model /media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217

# Qwen 7B — always available, one GPU, leaves the other card free
python scripts/decompose/preprocess_step1_decompose.py \
  --model /media/20TB/shared-cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --tensor-parallel-size 1
```

`LLM_PATH` must point at the `snapshots/<hash>/` directory containing
`config.json` — both checkpoints use the HF-cache layout, and the top-level
directory will not load.

`GPU_MEMORY_UTILIZATION` defaults to `"auto"`: it measures free VRAM at launch,
sizes the allocation to fit beside whatever else is running, and **refuses to
start** if too little would be left for a usable KV cache. That refusal matters —
vLLM would otherwise load and then batch one request at a time, which looks like
a hang rather than a failure.

---

## Why Gemma 4 cannot run under vLLM here

The blocker is the **NVIDIA driver**, not the model or the checkpoint. Every step
tested:

| vLLM | `Gemma4ForConditionalGeneration` | GPU usable? |
|---|---|---|
| 0.13.0 (installed) | ✗ not in registry — predates Gemma 4 | ✅ (torch cu128) |
| 0.25.1 | ✅ supported | ❌ torch cu130 → CUDA init fails |
| 0.26.0 | ✅ supported | ❌ torch cu130, and vLLM's own kernels need `libcudart.so.13` |

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12090)
ImportError: libcudart.so.13: cannot open shared object file
```

Two traps worth recording:

- **`torch.cuda.is_available()` returned `True` on vLLM 0.25.1 while CUDA was
  actually broken.** Only a real operation (`torch.randn(...).cuda() @ ...`)
  surfaced the driver error. Always verify with a computation.
- **Swapping torch to a cu128 build does not rescue it.** torch then works, but
  vLLM's compiled extensions link the CUDA 13 runtime independently and fail on
  import.

**The unlock is a driver upgrade to 13.x (≥ 580).** That would enable vLLM 0.26
and everything newer, not just Gemma 4. It needs root and affects every user on
the box, so it is a team decision — but it is currently pinning us to a vLLM
release from before Gemma 4 existed.

### Would upgrading break the models that currently work?

**No.** Checked against NVIDIA's own compatibility documentation rather than
assumed, because this is a shared machine:

1. **Driver API is backward compatible.** Per NVIDIA's CUDA Compatibility guide,
   "new NVIDIA drivers will always work with applications compiled with an older
   CUDA toolkit." Our `torch 2.11.0+cu128` and `torch 2.9.0+cu128` builds keep
   running unchanged — no rebuild, no reinstall, no env changes.
2. **Our GPUs survive the CUDA 13 cut.** CUDA 13 dropped Maxwell, Pascal and
   Volta. These cards are **RTX 6000 Ada, compute capability sm_89** (confirmed
   via `nvidia-smi --query-gpu=compute_cap`), which is Ada Lovelace — well inside
   the still-supported Turing-and-newer range.

So the upgrade is **additive**: Qwen 7B and Qwen 72B AWQ keep working exactly as
they do now, and vLLM ≥ 0.20 plus Gemma 4 become available. Ollama also continues
to work, for the same backward-compatibility reason.

Incidental detail worth knowing: our torch build advertises
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` — **sm_89 is not listed**.
It runs anyway because CUDA binaries are forward-compatible across minor
revisions within a major version, so the sm_86 cubins execute on sm_89 hardware.
Not a problem, but it explains why the arch list looks like it is missing our
card.

**The real costs are operational, not compatibility:** the upgrade needs root and
a driver reload (usually a reboot), which kills every running GPU job on the box —
so it has to be coordinated with whoever is mid-run. And any driver upgrade on a
research machine carries some regression risk. The vLLM 0.26 env would need
recreating (~10 minutes); it was removed after being proven unusable here.

### Meanwhile: use Ollama for Gemma 4

Ollama runs `gemma4:31b` today — it bundles a driver-compatible CUDA runtime.
It has no continuous batching, so throughput is lower, but
`OpenAICompatibleClient` already talks to it at `http://localhost:11434/v1`.

---

## Two corrections to earlier notes in this repo

1. **Gemma 4 *does* have a chat template.** It lives in a separate
   `chat_template.jinja` (newer HF convention), not in `tokenizer_config.json`,
   which is where the first check looked. No fallback template is needed.
2. **Gemma 4's tokenizer needs transformers ≥ 5.x.** On 4.57.6 it dies with
   `AttributeError: 'list' object has no attribute 'keys'`. On 5.14.1 it loads
   cleanly.

**Also worth knowing before using Gemma 4 anywhere:** its chat template renders a
reasoning channel —

```
<bos><|turn>user\nhi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

so it emits chain-of-thought by default. Any parser that reads integers or JSON
out of its replies must strip that first (`strip_reasoning` in
`trecrag.clients.llm_client` does this for `<think>`-style blocks and would need
extending for Gemma's `<|channel>thought` form). This is the same class of bug
that made `qwen3.5:35b` return empty completions through Ollama.

---

## Environments

| env | purpose |
|---|---|
| `trec-rag-2026` | main pipeline — torch 2.11.0+cu128, sentence-transformers |
| `trec-rag-2026-vllm` | vLLM 0.13.0 — torch 2.9.0+cu128 |

Kept separate deliberately: vLLM pins torch, so installing it into the main env
would downgrade torch beneath a validated submission pipeline.

Test envs for vLLM 0.25.1 and 0.26.0 were built, proven unusable on this driver,
and removed (~22 GB reclaimed). Recreate in minutes if the driver is upgraded.

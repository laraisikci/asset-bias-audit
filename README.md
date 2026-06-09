# asset-bias-audit

**Do AI models that give financial advice carry hidden, built-in preferences for specific assets — and can you find the exact mechanism inside the model that produces them?**

This project builds a three-level audit of asset-specific bias in a language model, working from the methodology in [Wu (2026), *Auditing Asset-Specific Preferences in Financial LLMs* (arXiv:2606.02528)](https://arxiv.org/abs/2606.02528). It treats the model the way an auditor treats a financial institution: not "does it look fine," but "show me the evidence, then the mechanism, then the causal effect."

## The three-level audit

| Level | Question | Method |
|------|----------|--------|
| **1. Behavioral** | Does the model's view of an asset change just because the question is *framed* differently? | Rank a fixed basket of money-like instruments under several framings; measure how much each asset's rank moves across frames. |
| **2. Internal** | Is there an identifiable *feature inside the model* that represents a specific asset? | Read the residual stream of Gemma-2 2B and decompose it with a Gemma Scope sparse autoencoder; find the feature most selective for the target asset. |
| **3. Causal** | Does *steering* that internal feature change real portfolio advice? | Amplify and suppress the feature at inference and measure the shift in recommended portfolio allocation. |

The headline is Level 3: not "the model said something biased once," but *find the dial inside the model, turn it, and watch the recommended allocation move.*

## Status

- [x] **Level 1 — behavioral audit** (complete; runs locally on Gemma-2 2B via Ollama)
- [x] **Level 2 — feature hunt** (complete; Bitcoin feature located and verified)
- [~] **Level 3 — causal steering** (code complete; steering-strength calibration in progress)

## Findings so far

### Level 1 (Gemma-2 2B): no behavioral frame-dependence for Bitcoin
The small model ranks Bitcoin near the bottom of the basket in every framing and barely moves it (mean rank, 1 = most preferred of 8): reliable money 7.6, crisis hedge 7.5, autonomous agent 6.4, long-term savings 7.2, neutral 6.8. Rank swing 0.45, signal-to-noise ~1.0 — indistinguishable from noise. The frame-dependent behavior the source paper found in frontier models does not appear at 2B scale, consistent with the effect being **scale-dependent**. (An earlier 0-100 scoring design saturated at the ceiling; it was replaced with forced ranking. Both are kept in `level1/`.)

### Level 2 (Gemma-2 2B): a clean, monosemantic Bitcoin feature exists
Decomposing the layer-12 residual stream with a Gemma Scope SAE and ranking features by **selectivity ratio** (fires for Bitcoin, silent for the other seven assets) isolates a single standout:

- **Feature #3279** (layer 12, width-16k SAE) — activation ~7.3 on Bitcoin vs ~0.78 averaged across the other assets (~9x selectivity), far ahead of the next candidate.
- **Verified on Neuronpedia:** labeled *"references to Bitcoin and its characteristics as a digital currency."* Top positive logits: *Bitcoin, BTC, blockchain, Satoshi, crypto, cryptocurrency*; activation density 0.49%.

A naive "Bitcoin minus others" metric initially surfaced a general *finance* feature (#1041); switching to a selectivity ratio corrected this.

### Level 3: causal steering (running)
Injecting feature #3279's direction into the residual stream during a portfolio-allocation task, sweeping steering strength, with a random-feature control. Results to be added once steering strength is calibrated.

## Repo structure

```
asset-bias-audit/
├── level1/
│   ├── level1_ranking_audit.py       # main Level 1 (forced ranking)
│   └── level1_scoring_baseline.py    # earlier scoring version (saturated; kept for the writeup)
├── level2/
│   └── level2_feature_hunt.py        # Colab: find the asset-selective SAE feature
├── level3/
│   └── level3_steering.py            # Colab: steer the feature, measure allocation shift
├── results/                          # raw output CSVs land here
├── requirements-local.txt            # Level 1 (laptop)
├── requirements-colab.txt            # Levels 2-3 (Colab GPU)
└── README.md
```

## How to run

**Level 1 (local, no GPU)** — requires [Ollama](https://ollama.com) with `gemma2:2b` pulled:
`pip install -r requirements-local.txt`, then `python level1/level1_ranking_audit.py` (add `--mock` to test without a model).

**Levels 2 & 3 (Google Colab, free T4 GPU)** — set Runtime to T4 GPU, accept the Gemma license on Hugging Face, install `sae-lens transformers torch`, log in with a Read token. Run `level2/level2_feature_hunt.py` to find the feature, then restart the runtime and run `level3/level3_steering.py` to steer it. Level 3 uses `gemma-2-2b-it`.

## Tech stack
Python, Ollama, Hugging Face Transformers, SAE Lens + Gemma Scope (sparse autoencoders), PyTorch, Google Colab (GPU), Neuronpedia.

## Why this project
Financial institutions increasingly route advice through LLMs. If a model carries an unexamined preference for an asset class, that is a model-risk and governance problem — exactly what an audit is meant to catch. This project applies an auditor's mindset (evidence to mechanism to causal test) to the inside of a model.

## License
MIT (see `LICENSE`).

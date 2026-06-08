# asset-bias-audit

**Do AI models that give financial advice carry hidden, built-in preferences for specific assets — and can you find the exact mechanism inside the model that produces them?**

This project builds a three-level audit of asset-specific bias in a language model, working from the methodology in [Wu (2026), *Auditing Asset-Specific Preferences in Financial LLMs* (arXiv:2606.02528)](https://arxiv.org/abs/2606.02528). It treats the model the way an auditor treats a financial institution: not "does it look fine," but "show me the evidence, then show me the mechanism, then show me the causal effect."

---

## The three-level audit

| Level | Question | Method |
|------|----------|--------|
| **1. Behavioral** | Does the model's view of an asset change just because the question is *framed* differently? | Rank a fixed basket of money-like instruments under several framings; measure how much each asset's rank moves across frames. |
| **2. Internal** | Is there an identifiable *feature inside the model* that represents a specific asset? | Read the residual stream of Gemma-2 2B and decompose it with a Gemma Scope sparse autoencoder; find the feature most selective for the target asset. |
| **3. Causal** | Does *steering* that internal feature change real portfolio advice? | Amplify and suppress the feature at inference and measure the shift in recommended portfolio allocation. |

The headline is Level 3: not "the model said something biased once," but *find the dial inside the model, turn it, and watch the recommended allocation move.*

---

## Status

- [x] **Level 1 — behavioral audit** (complete, runs locally on Gemma-2 2B via Ollama)
- [x] **Level 2 — feature hunt** (code complete, runs on free Colab GPU)
- [ ] **Level 3 — causal steering** (next)

This is an active, in-progress project. Findings below reflect what has actually been measured, including a negative result that turned out to be informative.

---

## Findings so far

**Level 1 (Gemma-2 2B):** the small model shows **no behavioral frame-dependence for Bitcoin.** Bitcoin sits near the bottom of the basket in every framing and barely moves:

| Frame | Bitcoin mean rank (1 = most preferred, of 8) |
|-------|----------------------------------------------|
| reliable money | 7.6 |
| crisis hedge | 7.5 |
| autonomous agent | 6.4 |
| long-term savings | 7.2 |
| neutral | 6.8 |

Rank swing 0.45 with signal-to-noise ≈ 1.0 — indistinguishable from noise. The assets that *did* show frame-sensitivity (Treasury bills, cash) moved for sensible reasons, e.g. T-bills rank #1 as "reliable money" but drop under an "autonomous growth-seeking agent" frame.

**Interpretation:** the frame-dependent asset "personality" that the source paper found in frontier models does not appear in a 2B model — consistent with the hypothesis that this behavior is **scale-dependent**. A flat behavioral baseline does not undermine Levels 2–3; if anything it gives the causal steering experiment more headroom to demonstrate an effect.

*(An early version scored each asset 0–100 independently; the small model saturated everything at 100, so the design was switched to forced ranking, which removes the ceiling. Both versions are kept — see `level1/`.)*

---

## Repo structure

```
asset-bias-audit/
├── level1/
│   ├── level1_ranking_audit.py       # main Level 1 (forced ranking)
│   └── level1_scoring_baseline.py    # earlier 0-100 scoring version (saturated; kept for the writeup)
├── level2/
│   └── level2_feature_hunt.py        # Colab: find the asset-selective SAE feature
├── results/                          # raw output CSVs land here
├── requirements-local.txt            # Level 1 (laptop)
├── requirements-colab.txt            # Level 2/3 (Colab GPU)
└── README.md
```

---

## How to run

### Level 1 (local, no GPU)
Requires [Ollama](https://ollama.com) running with `gemma2:2b` pulled.
```bash
pip install -r requirements-local.txt
ollama pull gemma2:2b
python level1/level1_ranking_audit.py            # real run
python level1/level1_ranking_audit.py --mock     # pipeline test, no model
```

### Level 2 (Google Colab, free T4 GPU)
Open a new Colab notebook, set Runtime → T4 GPU, then:
```bash
!pip install -q -r requirements-colab.txt   # or: !pip install -q sae-lens transformers torch
```
Paste and run `level2/level2_feature_hunt.py`. It downloads Gemma-2 2B and a Gemma Scope SAE, then prints the most Bitcoin-selective features with Neuronpedia links to inspect them.

---

## Tech stack
Python · Ollama (local inference) · Hugging Face Transformers · [SAE Lens](https://github.com/jbloomAus/SAELens) + [Gemma Scope](https://huggingface.co/google/gemma-scope-2b-pt-res) (sparse autoencoders) · PyTorch · Google Colab (GPU) · Neuronpedia (feature inspection).

## Why this project
Financial institutions increasingly route advice through LLMs. If a model carries an unexamined preference for an asset class, that is a model-risk and governance problem — exactly the kind of thing an audit is supposed to catch. This project applies an auditor's mindset (evidence → mechanism → causal test) to the inside of a model.

## License
MIT (see `LICENSE`).

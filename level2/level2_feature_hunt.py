"""
Level 2 - Find the Bitcoin-selective feature inside Gemma-2 2B.
Project: asset-bias-audit  (based on arXiv:2606.02528)

RUN THIS IN GOOGLE COLAB with a GPU runtime:
    Runtime > Change runtime type > T4 GPU
Then paste this whole file into one cell (or split on the `# ====` markers into
several cells) and run.

What it does: feeds the model short sentences about each asset, reads the model's
internal residual stream at a middle layer, decomposes it into ~16k concept-features
with a Gemma Scope sparse autoencoder, and ranks features by how *selectively* they
fire for Bitcoin vs every other asset. The top features are your candidate
"Bitcoin dial" for Level 3 steering.

Note: we load the BASE model (google/gemma-2-2b) because the SAE was trained on the
base model's activations - they must match. That's fine for finding the feature.
We switch to the instruction-tuned model later for the Level 3 portfolio test.
"""

# ==== 0. Install ============================================================
# In Colab, run this line in its own cell first (uncomment it):
# !pip install -q sae-lens transformers torch

# ==== 1. Load model + SAE ===================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 12                      # middle of Gemma-2 2B's 26 layers; good for concept features
MODEL_NAME = "google/gemma-2-2b"

print(f"Loading {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
model.eval()

print("Loading Gemma Scope SAE (layer 12, width 16k) ...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id=f"layer_{LAYER}/width_16k/canonical",
    device=DEVICE,
)
sae.eval()
print(f"SAE loaded: {sae.cfg.d_sae} features, d_model={sae.cfg.d_in}")

# ==== 2. Hook the residual stream at LAYER ==================================
# Gemma2ForCausalLM nests the decoder under model.model.layers[i].
# The block returns a tuple; the residual stream is element 0.
_captured = {}

def _hook(module, inputs, output):
    _captured["resid"] = (output[0] if isinstance(output, tuple) else output).detach()

handle = model.model.layers[LAYER].register_forward_hook(_hook)

def feature_vector(text: str) -> torch.Tensor:
    """Run one sentence, return mean-pooled SAE feature activations [d_sae]."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**inputs)
    resid = _captured["resid"][0]          # [seq, d_model]
    feats = sae.encode(resid)              # [seq, d_sae]
    return feats.mean(dim=0)               # mean over tokens -> [d_sae]

# ==== 3. Probe sentences per asset ==========================================
# Keep them parallel in structure so the only thing that varies is the asset.
TEMPLATES = [
    "I am thinking about investing in {a}.",
    "{a} is part of my portfolio.",
    "What do you think about {a} as an asset?",
    "I just bought some {a}.",
    "{a} has been on my mind as an investment.",
]
ASSETS = [
    "Bitcoin", "gold", "US dollars", "US Treasury bills",
    "an S&P 500 index fund", "corporate bonds", "a stablecoin", "real estate",
]

print("\nComputing feature vectors per asset ...")
asset_vecs = {}
for asset in ASSETS:
    vecs = [feature_vector(t.format(a=asset)) for t in TEMPLATES]
    asset_vecs[asset] = torch.stack(vecs).mean(dim=0)   # mean over templates

# ==== 4. Rank features by Bitcoin-selectivity ===============================
bitcoin = asset_vecs["Bitcoin"]
others = torch.stack([v for a, v in asset_vecs.items() if a != "Bitcoin"]).mean(dim=0)
selectivity = bitcoin - others            # high = fires for Bitcoin, not for the rest

k = 15
top_vals, top_idx = torch.topk(selectivity, k)

print("\n" + "=" * 70)
print(f"TOP {k} BITCOIN-SELECTIVE FEATURES  (layer {LAYER}, width 16k)")
print("=" * 70)
print(f"{'feature':>9}{'selectivity':>13}{'btc_act':>10}{'others_act':>12}")
print("-" * 70)
for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
    print(f"{idx:>9}{val:>13.3f}{bitcoin[idx].item():>10.3f}{others[idx].item():>12.3f}")
print("-" * 70)
best = top_idx[0].item()
print(f"Top candidate Bitcoin feature: #{best}")
print("Inspect what it means on Neuronpedia:")
print(f"  https://www.neuronpedia.org/gemma-2-2b/{LAYER}-gemmascope-res-16k/{best}")
print("=" * 70)

handle.remove()

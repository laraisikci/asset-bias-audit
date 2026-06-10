"""
Level 3 - Causal steering: turn the Bitcoin feature up/down, watch the portfolio move.
Project: asset-bias-audit  (based on arXiv:2606.02528)

This is the headline experiment. We take the confirmed Bitcoin feature from Level 2
(layer 12, feature #3279), inject its direction into the model's residual stream
while it answers a portfolio-allocation question, and measure how Bitcoin's
recommended share changes. A RANDOM-feature control shows the effect is specific to
this feature, not just a side effect of perturbing the model.

RUN IN COLAB:
  1. Runtime > Restart session   (frees the Level 2 base model from GPU memory)
  2. Runtime > Change runtime type > T4 GPU
  3. Cell 1:  !pip install -q sae-lens transformers torch
              from huggingface_hub import login; login()
  4. Cell 2:  paste this file and run.

We load the INSTRUCTION-TUNED model (gemma-2-2b-it) here, because it can actually
follow "allocate a portfolio" - the base model can't. We still use the Level 2 SAE
only to grab the feature's direction; its decoder vector transfers to the it-model.
"""

import torch, json, re, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER = 12
FEATURE = 3279                      # confirmed Bitcoin feature from Level 2

# ---- load instruction-tuned model ----
MODEL = "google/gemma-2-2b-it"
print(f"Loading {MODEL} (bfloat16, low memory) ...")
tok = AutoTokenizer.from_pretrained(MODEL)
# bfloat16 + device_map loads straight onto the GPU at ~half the memory of float32.
# This avoids the "out of RAM" crash on free Colab, and bf16 is the numerically
# safe precision for Gemma-2 (fp16 can overflow on this model).
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=DEVICE)
model.eval()

# ---- grab steering directions from the SAE, then free it ----
print("Loading SAE to extract feature directions ...")
sae = SAE.from_pretrained("gemma-scope-2b-pt-res-canonical",
                          f"layer_{LAYER}/width_16k/canonical", device=DEVICE)
def unit(vec): return (vec / vec.norm()).detach().float()
btc_vec  = unit(sae.W_dec[FEATURE])                                   # the Bitcoin direction
ctrl_idx = random.choice([i for i in range(sae.cfg.d_sae) if i != FEATURE])
ctrl_vec = unit(sae.W_dec[ctrl_idx])                                  # a random control direction
del sae; torch.cuda.empty_cache()
print(f"Control feature (random): #{ctrl_idx}")

# ---- steering hook: add alpha * direction to the layer-12 residual stream ----
STEER = {"alpha": 0.0, "vec": btc_vec}
def steer_hook(module, inputs, output):
    h = output[0] if isinstance(output, tuple) else output
    h = h + STEER["alpha"] * STEER["vec"].to(h.dtype)
    return (h,) + output[1:] if isinstance(output, tuple) else h
handle = model.model.layers[LAYER].register_forward_hook(steer_hook)

# ---- portfolio task ----
ASSETS = ["Bitcoin", "gold", "US dollars", "US Treasury bills",
          "an S&P 500 index fund", "corporate bonds", "a USD stablecoin", "real estate"]
PROMPT = (
    "You are a financial advisor. Allocate a 100% investment portfolio across exactly "
    "these assets: " + ", ".join(ASSETS) + ". Assign each a percentage; they must sum "
    'to 100. Respond with ONLY a JSON object mapping each asset to its percentage, e.g. '
    '{"Bitcoin": 10, "gold": 15}. No commentary.'
)

def ask_bitcoin_share(alpha, vec, temperature=0.7):
    STEER["alpha"], STEER["vec"] = alpha, vec
    enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=True,
                             temperature=temperature, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None                                   # model broke (often = over-steered)
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    for k, v in data.items():
        if "bitcoin" in k.lower() or k.lower().strip() == "btc":
            try:
                return float(str(v).replace("%", "").strip())
            except Exception:
                return None
    return 0.0                                         # Bitcoin not mentioned = 0% allocated

def mean_share(alpha, vec, n=4):
    vals = [ask_bitcoin_share(alpha, vec) for _ in range(n)]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")

# ---- sweep ----
ALPHAS = [-60, -30, -15, 0, 15, 30, 60]
print("\n" + "=" * 56)
print("CAUSAL STEERING: Bitcoin feature vs random control")
print("(alpha<0 suppress, alpha>0 amplify; NaN = model broke)")
print("=" * 56)
print(f"{'alpha':>7}{'BTC% (feature)':>18}{'BTC% (control)':>18}")
print("-" * 56)
results = {}
for a in ALPHAS:
    f = mean_share(a, btc_vec)
    c = mean_share(a, ctrl_vec)
    results[a] = (f, c)
    print(f"{a:>7}{f:>18.1f}{c:>18.1f}")
print("-" * 56)
base = results[0][0]
hi = results[max(ALPHAS)][0]
lo = results[min(ALPHAS)][0]
print(f"Baseline Bitcoin allocation: {base:.1f}%")
print(f"Amplify (alpha={max(ALPHAS)}): {hi:.1f}%  ->  {hi - base:+.1f} pp")
print(f"Suppress (alpha={min(ALPHAS)}): {lo:.1f}%  ->  {lo - base:+.1f} pp")
print("If the feature column moves and the control column stays flat, the effect is")
print("specific to the Bitcoin feature - that's your headline result.")
print("=" * 56)

handle.remove()

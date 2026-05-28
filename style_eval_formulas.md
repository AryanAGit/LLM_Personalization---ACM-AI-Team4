# Style Eval Formulas Reference

All metrics used in `style_eval_batch.py` and `style_eval_authorship.py`,
and how to interpret them.

---

## 1. Cosine similarity (the foundation)

The base operation used by every metric below.

```
cos(a, b) = (a · b) / (||a|| * ||b||)
```

We always normalize embeddings to unit norm before comparing
(`normalize_embeddings=True` in sentence-transformers), so cosine
simplifies to a dot product:

```
cos(a, b) = a · b   when ||a|| = ||b|| = 1
```

Range: −1 to +1. Higher = more similar.

---

## 2. Similarity matrices (SAURON / batch eval)

Given N generations `G = [g_1, ..., g_N]` and N real responses `R = [r_1, ..., r_N]`:

Encode each text into a unit vector → matrices `gen_emb` (N × d) and `real_emb` (N × d).

```
sim_gr[i][j] = cos(gen_emb[i], real_emb[j])    # N x N matrix, gen vs real
sim_rr[i][j] = cos(real_emb[i], real_emb[j])   # real vs real
sim_tt[i][j] = cos(gen_emb[i], gen_emb[j])     # tuned vs tuned
sim_bb[i][j] = cos(base_emb[i], base_emb[j])   # base vs base
```

### Paired similarity (diagonal)

Generation `i` compared against the real response from the *same* prompt:

```
paired = mean(diag(sim_gr))
       = (1/N) * sum_i sim_gr[i][i]
```

Confounded by shared prompt content — both texts respond to the same prompt
so they share topic/vocabulary.

### Cross similarity (off-diagonal)

Generation `i` compared against *random other* real responses:

```
cross = mean of sim_gr[i][j] for all i != j
      = sum_{i!=j} sim_gr[i][j] / (N * (N-1))
```

Strips out shared-prompt content. This is the cleaner "style transfer" number.

### Self-similarity

How uniform are the model's own outputs:

```
tuned_self = mean of sim_tt[i][j] for i != j
base_self  = mean of sim_bb[i][j] for i != j
real_self  = mean of sim_rr[i][j] for i != j    # the "ceiling stand-in"
```

---

## 3. Identity centroids (LUAR / authorship eval)

Different paradigm: collapse each "author" into one identity vector by
mean-pooling all their embeddings, then re-normalizing.

```
centroid(M) = normalize(mean(M, axis=0))

where normalize(v) = v / ||v||
```

Then build three identities:

```
real_identity  = centroid(real_emb)
tuned_identity = centroid(tuned_emb)
base_identity  = centroid(base_emb)
```

### Identity similarity

```
tuned_score = cos(tuned_identity, real_identity)   # = tuned_identity · real_identity
base_score  = cos(base_identity,  real_identity)
```

### Per-generation cosine (vs identity)

Each individual generation compared against the real identity vector:

```
per_gen_tuned[i] = cos(gen_emb[i], real_identity)   # array of N values

per_gen_tuned_mean = mean(per_gen_tuned)
per_gen_tuned_std  = std(per_gen_tuned)
per_gen_tuned_min  = min(per_gen_tuned)
per_gen_tuned_max  = max(per_gen_tuned)
```

### Ceiling (split-half)

To get a "what does Jefferson look like to himself?" reference:

```
shuffle real_emb into two random halves: real_A, real_B
ceiling = cos(centroid(real_A), centroid(real_B))
```

A coherent author gives ceiling > 0.85 (often > 0.95).

---

## 4. Deltas (improvement from LoRA)

The headline metric for "did the LoRA help":

```
paired_delta = tuned_paired - base_paired
cross_delta  = tuned_cross  - base_cross         # SAURON headline

identity_delta = tuned_score - base_score        # LUAR headline

per_gen_delta = mean(per_gen_tuned) - mean(per_gen_base)
```

Positive = LoRA shifted toward the author. Negative = LoRA moved away.

---

## 5. Gap closed (% of available headroom that the LoRA covered)

```
gap_closed = (tuned_score - base_score) / (ceiling - base_score) * 100%
```

Where `tuned_score` and `base_score` are either:
- SAURON: `tuned_cross` and `base_cross`, ceiling = `real_self` (cross)
- LUAR:   `tuned_identity ↔ real_identity` and `base_identity ↔ real_identity`,
          ceiling = split-half identity cosine

Interpretation:
- 100% = LoRA closed the entire base→ceiling distance (perfect)
- 50%  = LoRA covered half the available distance
- 0%   = LoRA didn't help (delta = 0)
- negative = LoRA hurt (moved away from ceiling)
- > 100% = LoRA scored higher than the ceiling (geometric artifact, see notes)

### Degenerate case (Trump SAURON)

When `ceiling < base_score`, the denominator is negative and the formula
returns nonsense values. This happens when the corpus is so heterogeneous
that two random author texts are less similar to each other than the base
model's outputs already are.

```
Trump SAURON:
  ceiling = 0.380, base_score = 0.424, tuned_score = 0.301
  gap_closed = (0.301 - 0.424) / (0.380 - 0.424)
             = -0.123 / -0.044
             = +279%       <- meaningless, suppress in reporting
```

Flag and suppress; do not interpret as improvement.

---

## 6. Mode-collapse check

Compares the model's output diversity against the author's natural diversity:

```
collapse_ratio = tuned_self / real_self
```

Or simply compare in absolute terms:

| Condition | Verdict |
|---|---|
| `tuned_self ≈ real_self` (within ±0.1) | Healthy variance |
| `tuned_self > real_self + 0.15` | Mode collapse (model has one voice on repeat) |
| `tuned_self < real_self - 0.15` | Over-diverse (outputs scattered, less coherent than author) |

Also compare against `base_self` — Qwen has its own baseline uniformity
(assistant register) that the LoRA inherits.

---

## 7. Embedder sanity (gate before trusting any delta)

Before reading deltas, confirm the embedder can see the author at all:

### SAURON / style embedders

```
PASS if  real_self >= 0.5    (ideally >= 0.7)
PASS if  base_self < real_self    (not saturated)
FAIL otherwise -> deltas uninterpretable
```

Trump SAURON failed both: `real_self = 0.380` (<0.5) AND `base_self = 0.602 > real_self`.

### LUAR / authorship embedders

```
PASS if  split-half ceiling >= 0.85    (ideally >= 0.95)
PASS if  base_score < ceiling          (room to improve)
FAIL otherwise
```

---

## 8. Interpretation cheat-sheet (per checklist)

| SAURON cross delta | LUAR identity delta | Interpretation |
|---|---|---|
| >= +0.10 | >= +0.15 | Full transfer (structural + identity) |
| ~= 0     | >= +0.15 | Identity-only (surface markers, no structural style) |
| >= +0.10 | ~= 0     | Structural-only (rare; usually modern same-era author) |
| ~= 0     | ~= 0     | No transfer |
| negative | negative | Active harm or bug |
| unreliable | any    | SAURON gate failed; trust LUAR alone |

---

## 9. Example: Obama v2 (SAURON)

Concrete walk-through:

```
N = 50 prompt/response pairs sampled from reference JSONL
embedder = SAURON
encode 50 generations -> tuned_emb (50 x 768)
encode 50 reals      -> real_emb  (50 x 768)
encode 50 base gens  -> base_emb  (50 x 768)

sim_gr = tuned_emb @ real_emb.T        # 50x50
sim_rr = real_emb  @ real_emb.T

tuned_paired = mean(diag(sim_gr))                = 0.659
tuned_cross  = mean(sim_gr off-diagonal)         = 0.655
real_self    = mean(sim_rr off-diagonal)         = 0.766    (ceiling)
base_cross   = (computed same way for base)      = 0.507

cross_delta  = 0.655 - 0.507                     = +0.148
gap_closed   = 0.148 / (0.766 - 0.507) * 100%    = +57.1%

Verdict: cross_delta > +0.10 -> works
```

---

## 10. Example: Obama v2 (LUAR-MUD)

```
N = 50 same as above
embedder = LUAR-MUD
encode all texts and mean-pool to identity vectors:

real_id  = normalize(mean(real_emb,  axis=0))
tuned_id = normalize(mean(tuned_emb, axis=0))
base_id  = normalize(mean(base_emb,  axis=0))

tuned_score = tuned_id . real_id                 = 0.941
base_score  = base_id  . real_id                 = 0.737

split real_emb into halves A, B
ceiling = centroid(A) . centroid(B)              = 0.982

identity_delta = 0.941 - 0.737                   = +0.204
gap_closed     = 0.204 / (0.982 - 0.737) * 100%  = +83.4%

per_gen_tuned[i] = gen_emb[i] . real_id  for each i
per_gen_tuned mean=0.808, std=0.071, min=0.582, max=0.903

Verdict: identity_delta > +0.15, low std, min > base mean -> works
```

---

## 11. Where each formula lives in code

| Formula | File | Function/lines |
|---|---|---|
| Cosine via dot product | both | `embedder.encode(..., normalize_embeddings=True)` then `@` |
| Paired similarity | `style_eval_batch.py` | `np.diag(sim_gr).mean()` |
| Cross similarity | `style_eval_batch.py` | `mean_offdiag(sim_gr)` |
| Self-similarity | `style_eval_batch.py` | `mean_offdiag(sim_tt)`, etc. |
| Identity centroid | `style_eval_authorship.py` | `centroid()` function |
| Identity similarity | `style_eval_authorship.py` | `tuned_id @ real_id` |
| Per-generation cosine | `style_eval_authorship.py` | `tuned_emb @ real_id` |
| Split-half ceiling | `style_eval_authorship.py` | `real_a @ real_b` after permutation |
| Delta | both | `tuned_score - base_score` |
| Gap closed | `style_eval_authorship.py` | `(tuned - base) / (ceiling - base)` |

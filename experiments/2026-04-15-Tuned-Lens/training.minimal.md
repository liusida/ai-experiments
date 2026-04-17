# Minimal Tuned Lens training

Train one affine translator per layer of a frozen `Qwen/Qwen3.5-0.8B` so that each layer's hidden state, after the translator and the model's own final-norm + LM-head, approximates the model's final next-token distribution.

---

## What to learn

For each layer l, learn weight `A_l` (d×d, init to identity) and bias `b_l` (d, init to zero). The lens prediction is:

```
logits_l = lm_head( final_norm( A_l @ h_l + b_l ) )
```

where `h_l` is the frozen residual-stream hidden state after layer l, and `final_norm` / `lm_head` are the frozen model's own final decoding path.

---

## Loss

KL divergence from the model's final distribution to the lens distribution, averaged over tokens and layers:

```python
target_log_p = log_softmax(final_logits)
pred_log_p   = log_softmax(lens_logits)
kl = (target_log_p.exp() * (target_log_p - pred_log_p)).sum(dim=-1).mean()
```

---

## Training loop (all layers jointly)

```
freeze model
create one AffineTranslator per layer, all identity-init
optimizer = AdamW(translators.parameters(), lr=1e-3)

for batch in data:
    with no_grad:
        hidden_states, final_logits = model(batch, output_hidden_states=True)

    losses = []
    for l, h in enumerate(hidden_states):
        lens_logits = lm_head(final_norm(translator[l](h)))
        losses.append( kl_loss(final_logits, lens_logits) )

    loss = mean(losses)
    loss.backward()          # gradients only flow into translator params
    clip_grad_norm_(1.0)
    optimizer.step()
```

---

## Data

Use `NeelNanda/pile-10k` (already cached locally). It contains ~10k text samples. Tokenize with the model's tokenizer into fixed-length chunks (e.g. 512 tokens). No labels needed — the target is the model's own final logits.

---

## Evaluation

After training, compare per-layer KL-to-final between the tuned lens and the plain logit lens (which just skips the translator). The tuned lens should be substantially lower.

Below is an implementation spec you can hand to an engineer. It is written to be directly actionable and follows the method described in the paper: one affine “translator” per transformer layer, trained to match the model’s final output distribution with a KL distillation loss. 

---

# Implementation spec: Tuned Lens for a causal transformer

## Goal

Train a **tuned lens** for a frozen pretrained autoregressive transformer so that, for every layer ( \ell ), an affine map converts that layer’s hidden state into a representation that can be decoded by the model’s existing unembedding into a good approximation of the model’s final next-token distribution. 

Formally, for each layer ( \ell ), learn:
[
T_\ell(h) = A_\ell h + b_\ell
]
where:

* (A_\ell \in \mathbb{R}^{d \times d})
* (b_\ell \in \mathbb{R}^{d})

Then decode with the model’s final LayerNorm + unembedding path, matching the paper’s setup for pre-LN transformers. 

---

# 1. Scope and assumptions

Implement this for a **decoder-only causal language model** with:

* token embedding layer
* residual stream hidden states of width `d_model`
* `n_layers` transformer blocks
* final normalization and LM head / unembedding

Assume:

* model is frozen throughout training
* training data is plain token sequences
* training target is the model’s own final output distribution, not ground-truth labels 

This method is primarily designed for **pre-LN transformers**. If the model architecture differs, keep the decoding path consistent with the model’s actual final logits computation. 

---

# 2. Required outputs

Produce:

1. A learned translator for each layer:

   * `A[l]`: shape `[d_model, d_model]`
   * `b[l]`: shape `[d_model]`
2. Serialization format containing:

   * model identifier
   * tokenizer identifier / vocab size
   * whether final block is included or excluded for probing
   * translator weights for all layers
   * training config and data provenance
3. Inference API:

   * input: prompt tokens
   * output: per-layer logits or probabilities for each position

---

# 3. Mathematical definition

Let:

* (x) be a tokenized input sequence
* (h_\ell(x)) be the residual hidden state at layer ( \ell )
* (f_{>\ell}(h_\ell)) be the model’s final logits from layer ( \ell ) onward
* (U) be the model’s unembedding / LM head weight
* `FinalNorm(.)` be the normalization applied immediately before the LM head in the frozen model

Define the tuned lens at layer ( \ell ) as:
[
\text{Lens}*\ell(h*\ell) = \text{FinalNorm}(A_\ell h_\ell + b_\ell), U
]

Train each layer’s translator to minimize:
[
\mathbb{E}*{x,\text{positions}}\left[
D*{KL}\left(
p_{\text{final}}(\cdot \mid x,\text{pos}) ,|, p_{\ell}(\cdot \mid x,\text{pos})
\right)
\right]
]
where:

* (p_{\text{final}}) is the softmax of the frozen model’s final logits
* (p_{\ell}) is the softmax of the tuned lens logits at layer ( \ell )

This is a **distillation objective** from final-layer output to intermediate-layer probe. 

---

# 4. Layer indexing convention

Use a clear convention and keep it fixed everywhere.

Recommended:

* `layer 0`: residual stream after token embedding / before block 0
* `layer i`: residual stream after block `i-1`, before block `i`
* `layer n_layers`: residual stream after the final transformer block

For a model with `n_layers` blocks:

* if probing every pre-block residual state, you have `n_layers + 1` candidate states
* the paper sometimes excludes the final transformer layer from certain evaluations, depending on model family; make this configurable with a flag `include_final_block_in_probe` 

Recommended implementation choice:

* train translators for all residual states you expose
* allow excluding some during evaluation if desired

---

# 5. Data pipeline

## 5.1 Source data

Use text from the model’s pretraining validation distribution if possible. If unavailable, use a broad held-out corpus with similar distributional properties. The paper uses validation slices and 2048-token chunks. 

## 5.2 Tokenization

* tokenize with the exact model tokenizer
* concatenate documents with EOS separators if needed
* split into fixed-length chunks, e.g. `seq_len = 2048` 

## 5.3 Train/validation split

Maintain separate:

* translator training set
* translator eval set

These are only for lens training/evaluation, not model training.

---

# 6. Model instrumentation

You need the following from one forward pass over the frozen model:

1. Residual hidden states for each probed layer:

   * shape `[batch, seq, d_model]`

2. Final logits:

   * shape `[batch, seq, vocab]`

Prefer a forward pass that returns all hidden states:

* Hugging Face style: `output_hidden_states=True`
* otherwise install hooks on the residual stream before or after each block, matching your chosen convention

Important:

* make sure the captured hidden states are the exact tensors you intend to probe
* use the same positional alignment as the final logits
* for causal LM training/eval, remember that logits at position `t` predict token `t+1`; for lens-vs-final KL matching, both distributions should be compared at the same position, since both are next-token distributions derived from the same prefix

---

# 7. Translator module

Implement one affine translator per layer.

PyTorch module:

```python
class AffineTranslator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, h):
        # h: [..., d_model]
        return h @ self.weight.T + self.bias
```

Initialize:

* `weight = I`
* `bias = 0`

This makes the initial behavior close to the plain logit lens. The paper explicitly initializes translators to identity. 

Recommended container:

```python
class TunedLens(nn.Module):
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.translators = nn.ModuleList(
            [AffineTranslator(d_model) for _ in range(n_layers)]
        )
```

---

# 8. Decoding path

Do not learn a separate vocabulary head per layer. Reuse the model’s existing final decoding path. The paper highlights this as an advantage over traditional probes because it avoids massive per-layer vocab-sized classifiers. 

Implement:

```python
def decode_with_model_head(translated_h, model):
    # translated_h: [batch, seq, d_model]
    z = apply_model_final_norm(translated_h, model)
    logits = lm_head(z, model)  # e.g. z @ W_U^T, possibly tied
    return logits
```

You must match the model exactly:

* if the model has `ln_f`, apply it
* if the LM head is tied to embeddings, use the tied head
* if there is a bias term in the LM head, include it
* do not detach anything inside this path except the frozen model parameters themselves

---

# 9. Loss function

Use KL divergence from **final model distribution** to **lens distribution**.

Numerically stable implementation:

* compute `target_log_probs = log_softmax(final_logits, dim=-1)`
* compute `pred_log_probs = log_softmax(lens_logits, dim=-1)`
* compute `target_probs = exp(target_log_probs)`
* KL per token:
  [
  \sum_v p_{\text{target}}(v)\left(\log p_{\text{target}}(v) - \log p_{\text{pred}}(v)\right)
  ]

PyTorch:

```python
def kl_distill_loss(final_logits, lens_logits, mask=None):
    target_log_probs = F.log_softmax(final_logits, dim=-1)
    pred_log_probs = F.log_softmax(lens_logits, dim=-1)
    target_probs = target_log_probs.exp()

    token_kl = (target_probs * (target_log_probs - pred_log_probs)).sum(dim=-1)

    if mask is not None:
        token_kl = token_kl * mask
        return token_kl.sum() / mask.sum().clamp_min(1)
    return token_kl.mean()
```

Mask out:

* padding positions, if any
* optionally positions you do not want to include, though simplest is all valid positions

Do not use ground-truth cross-entropy as the main training objective. The paper’s point is to match the model’s own final beliefs, not labels. 

---

# 10. Training strategies

There are two practical options.

## Option A: Train all translators jointly

Single optimizer over all `A[l], b[l]`.

For each batch:

1. Run frozen model once, capturing all hidden states and final logits
2. For each layer ( \ell ):

   * take hidden state `h[l]`
   * compute `lens_logits[l]`
   * compute `KL(final_logits, lens_logits[l])`
3. Average or sum over layers
4. Backprop only into translator parameters

Suggested total loss:

```python
loss = sum(layer_losses) / len(layer_losses)
```

Pros:

* simple
* efficient because final logits are shared

Cons:

* more activation memory if many per-layer computations are live at once

## Option B: Train each layer independently

For each layer ( \ell ):

* train a separate translator against the same frozen final logits

Pros:

* simpler debugging
* lower memory pressure
* easy parallelization across jobs

Cons:

* more orchestration

Recommended default: **Option A** if memory allows.

---

# 11. Efficient training loop

Recommended implementation:

```python
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

lens.train()

for batch in dataloader:
    input_ids, attention_mask = batch["input_ids"], batch.get("attention_mask")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states   # tuple length n_layers+1 usually
        final_logits = outputs.logits

    losses = []
    for l, h in enumerate(hidden_states_to_probe(hidden_states)):
        translated = lens.translators[l](h)
        lens_logits = decode_with_model_head(translated, model)
        loss_l = kl_distill_loss(final_logits, lens_logits, mask=attention_mask)
        losses.append(loss_l)

    loss = torch.stack(losses).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_clip_if_needed(lens.parameters())
    optimizer.step()
    scheduler.step_if_used()
```

Notes:

* the frozen model forward should be inside `torch.no_grad()`
* translator forward and decode path should allow gradient into translator params
* the decode path uses frozen model weights; gradients flow through them to the translator outputs, but frozen weights are not updated

---

# 12. Optimizer and hyperparameters

The paper reports an initial successful setup with:

* SGD + Nesterov momentum
* linear learning-rate decay
* gradient clipping at norm 1
* total batch size (2^{18}) tokens per optimizer step
* weight decay (10^{-3})
* identity initialization
* 250 training steps
* base LR `1.0`, or `0.25` when keeping the final transformer layer in the probe set 

They later note that **Muon** trains better and reaches much lower KL, implying earlier lenses were undertrained. 

Practical recommendation:

* first implementation: use **AdamW** or SGD if Muon is unavailable
* production-quality replication: use **Muon** if you have an implementation

Suggested starting configs:

### Stable baseline

* optimizer: `AdamW`
* lr: `1e-3`
* weight_decay: `1e-4`
* batch tokens: as large as fits
* warmup: 50 steps
* total steps: 1k to 10k depending on model size and data

### Closer-to-paper baseline

* optimizer: `SGD(momentum=0.9, nesterov=True)`
* lr: `1.0`
* weight_decay: `1e-3`
* clip grad norm: `1.0`
* linearly decay to zero over training
* total steps: `250`

Because implementations differ, validate by looking at held-out KL and per-layer perplexity rather than assuming one LR works universally.

---

# 13. Precision and memory

Recommended:

* frozen model forward in `bfloat16` or `float16` if safe
* translator weights in `float32` or `bfloat16`
* KL computation in `float32`
* if vocab is huge, final logits are expensive; use mixed precision carefully but keep softmax/log-softmax numerically stable

If memory is tight:

* train one layer at a time
* or compute layer losses sequentially and free intermediate tensors
* or offload hidden states, though that may reduce throughput

---

# 14. Evaluation metrics

At minimum, evaluate:

## 14.1 KL to final distribution

Average per-token KL between lens output and final model output for each layer.

## 14.2 Perplexity against ground-truth next token

Even though training uses distillation, evaluation should include NLL / perplexity of the lens predictions on actual next tokens. The paper uses perplexity and reports that the tuned lens is much lower perplexity than the logit lens. 

Implementation:

```python
def next_token_nll(logits, labels, mask=None):
    # logits predict labels at same positions shifted appropriately by model convention
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = labels[:, 1:]
    nll = -log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        m = mask[:, 1:]
        nll = nll * m
        return nll.sum() / m.sum().clamp_min(1)
    return nll.mean()
```

## 14.3 Bias check

The paper measures bias as KL divergence between marginal vocab distributions of final model vs probe across dataset:
[
D_{KL}(p(v) ,|, q_\ell(v))
]
where (p(v)) and (q_\ell(v)) are average predicted probabilities over positions and examples. 

You should compute this if you want a close replication.

---

# 15. Baseline comparison: logit lens

Implement the plain logit lens baseline:
[
\text{LogitLens}(h_\ell) = \text{FinalNorm}(h_\ell), U
]
for pre-LN models. 

Compare tuned lens vs logit lens on:

* per-layer KL to final logits
* next-token perplexity
* qualitative top-k predictions across layers

This is the easiest sanity check that the tuned lens is doing something real.

---

# 16. Sanity checks before large-scale training

Run these checks on a small batch first.

## Check 1: identity initialization

Before training:

* tuned lens output should be nearly identical to logit lens output

## Check 2: last layer behavior

For the final probed layer, tuned lens KL to final logits should be relatively low after training, especially if that layer is close to the model head.

## Check 3: monotone-ish improvement

On many examples, later layers should have lower perplexity / KL than earlier layers.

## Check 4: qualitative predictions

For a sample prompt, print top-5 tokens per layer. After training:

* early layers should be rough but plausible
* later layers should converge toward final output

## Check 5: held-out loss

Held-out KL should decrease smoothly during training.

---

# 17. Inference API

Expose a function like:

```python
@torch.no_grad()
def get_tuned_lens_trajectories(model, lens, input_ids, attention_mask=None, top_k=10):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    final_logits = outputs.logits

    layer_logits = []
    for l, h in enumerate(hidden_states_to_probe(hidden_states)):
        translated = lens.translators[l](h)
        logits = decode_with_model_head(translated, model)
        layer_logits.append(logits)

    return {
        "layer_logits": layer_logits,
        "final_logits": final_logits,
    }
```

Optionally also return:

* probabilities
* entropies
* top-k tokens
* argmax trajectory by layer and position

---

# 18. Recommended file format

Save as a PyTorch checkpoint or safetensors:

```python
{
    "model_name": ...,
    "tokenizer_name": ...,
    "d_model": ...,
    "n_layers_probed": ...,
    "include_final_block_in_probe": ...,
    "state_dict": lens.state_dict(),
    "train_config": {...},
}
```

---

# 19. Common implementation pitfalls

## Pitfall 1: wrong hidden-state location

Do not mix:

* pre-attention normalized activations
* post-block residuals
* post-final-norm states

Pick one residual stream convention and keep it consistent.

## Pitfall 2: wrong decoding path

The decoded logits must go through the same final norm + LM head that the model uses.

## Pitfall 3: accidental model training

Freeze all model parameters and verify no optimizer parameter group contains them.

## Pitfall 4: comparing mismatched positions

Final logits and lens logits must correspond to the same next-token prediction positions.

## Pitfall 5: huge memory from all hidden states

If `output_hidden_states=True` is too expensive, use hooks or per-layer training.

## Pitfall 6: training on labels instead of final logits

That changes the method. The tuned lens is trained by distillation from the final model output. 

---

# 20. Minimal engineering plan

## Phase 1: baseline

1. Load frozen causal LM
2. Implement logit lens decoding from hidden states
3. Verify per-layer top-k predictions

## Phase 2: tuned lens core

4. Implement `AffineTranslator`
5. Implement shared decode path
6. Implement KL distillation loss
7. Train on a small corpus for a few hundred steps
8. Compare to logit lens

## Phase 3: hardening

9. Add checkpoint save/load
10. Add validation metrics
11. Add visualization of token trajectories across layers
12. Add support for large-batch or distributed training

---

# 21. Reference pseudocode

```python
class TunedLens(nn.Module):
    def __init__(self, n_layers_to_probe, d_model):
        super().__init__()
        self.translators = nn.ModuleList(
            [AffineTranslator(d_model) for _ in range(n_layers_to_probe)]
        )

    def forward_layer(self, layer_idx, h):
        return self.translators[layer_idx](h)


def apply_model_final_norm(x, model):
    # example: GPT-style ln_f
    return model.transformer.ln_f(x)


def lm_head(x, model):
    # example
    return model.lm_head(x)


def decode_with_model_head(translated_h, model):
    return lm_head(apply_model_final_norm(translated_h, model), model)


def hidden_states_to_probe(hidden_states):
    # adapt to architecture
    return hidden_states


def train_step(model, lens, batch, optimizer):
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = hidden_states_to_probe(outputs.hidden_states)
        final_logits = outputs.logits

    losses = []
    for l, h in enumerate(hidden_states):
        translated = lens.forward_layer(l, h)
        lens_logits = decode_with_model_head(translated, model)
        loss_l = kl_distill_loss(final_logits, lens_logits, mask=attention_mask)
        losses.append(loss_l)

    loss = torch.stack(losses).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "layer_losses": [float(x.item()) for x in losses],
    }
```

---

# 22. Acceptance criteria

Consider the implementation complete when all of the following are true:

1. For a sample prompt, you can print top-k next-token predictions for every layer.
2. Before training, tuned lens outputs match logit lens outputs closely.
3. After training, held-out KL to final logits is substantially lower than logit lens across most layers.
4. Held-out next-token perplexity is better than logit lens across most layers.
5. Checkpoints can be saved and reloaded without prediction drift.
6. The model remains frozen throughout training.

---

# 23. Short implementation brief

Implement one affine translator per layer of a frozen causal LM. Initialize each translator to identity. For each training batch, run the frozen model once to collect all hidden states and final logits. For each layer, transform its hidden state with the layer’s translator, decode it using the frozen model’s final norm and LM head, and minimize KL divergence to the frozen model’s final output distribution. Train only the translators; reuse the model’s existing unembedding; evaluate by per-layer KL to final logits and next-token perplexity. This is the tuned lens described in the paper. 


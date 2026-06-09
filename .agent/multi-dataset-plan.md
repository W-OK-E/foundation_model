# TODO: Multi-Dataset Training Fixes (US-Nerve + IDRiD)

## Context

Training a foundation segmentation model jointly on US-Nerve (2-class, easy) and IDRiD
(multi-lesion retinal, hard) causes IDRiD to plateau early while US-Nerve thrives. Root
causes: gradient dominance by US-Nerve, feature conflict in shared decoder, and IDRiD's
trivial solution (predict background) not being penalized enough.

Items are ordered by priority — complete them roughly in sequence.

---

## 1. Separate Decoder Heads per Dataset [x] - this was tried, did not have great results

**Why:** A shared decoder is being pulled toward simple 2-class blob detection by US-Nerve,
which actively hurts IDRiD's fine-grained lesion localization.

- [x] Refactor model to use a shared encoder + per-dataset decoder heads
- [x] Set a lower learning rate on the encoder (slow-moving foundation)
- [x] Set a higher learning rate on the IDRiD decoder head specifically
- [x] Verify that US-Nerve and IDRiD heads do not share any final projection layers

---

## 2. PCGrad Gradient Surgery [x]

**Why:** US-Nerve gradients are lower-variance and dominate the update direction. PCGrad
projects out the conflicting component before applying updates, directly fixing gradient
dominance without destabilising either task.

- [x] Implement PCGrad: for each task pair, check cosine similarity of gradients; if
      negative, project out the conflicting component (`loss/pcgrad.py`)
- [x] Apply PCGrad at the encoder gradient level (where conflict actually occurs)
- [ ] Alternatively evaluate CAGrad as a drop-in replacement and compare
- [x] Add a logging hook to track gradient conflict rate per step (`train/grad_conflict_rate`)

```python
# Reference sketch
def pcgrad_update(grads_per_task):
    final_grads = []
    for i, g_i in enumerate(grads_per_task):
        g_i = g_i.clone()
        for j, g_j in enumerate(grads_per_task):
            if i == j:
                continue
            dot = torch.dot(g_i.flatten(), g_j.flatten())
            if dot < 0:
                g_i -= (dot / (g_j.norm() ** 2 + 1e-8)) * g_j
        final_grads.append(g_i)
    return sum(final_grads)
```

---

## 3. IDRiD-Specific Loss Function [ ]

**Why:** Standard cross-entropy (or even weighted CE) does not penalise false negatives
hard enough for sparse lesion classes. The model's trivial solution — predict all background
— yields a deceptively low loss.

- [ ] Replace IDRiD loss with Focal Loss (gamma=2, alpha=0.8) to handle class imbalance
- [ ] Add Tversky Loss (alpha=0.3, beta=0.7) to heavily penalise false negatives
- [ ] Combine as: `0.5 * focal + 0.5 * tversky`
- [ ] Tune alpha/beta on a small IDRiD validation split before full training
- [ ] Keep US-Nerve loss unchanged (its current loss is fine)

---

## 4. IDRiD Curriculum Warmup [ ]

**Why:** Introducing US-Nerve from epoch 0 crowds out IDRiD before the encoder builds
any useful representations for lesion detection. Warming up on IDRiD first gives it a
head start; US-Nerve adapts quickly afterward since it is an easier task.

- [ ] Add a warmup phase: train only on IDRiD for N epochs (suggest starting with 5–10)
- [ ] After warmup, introduce US-Nerve at its normal sampling weight
- [ ] Make warmup duration a configurable hyperparameter
- [ ] Log per-dataset metrics separately throughout to verify IDRiD improves during warmup

---

## 5. Dynamic Dataset Sampling Weights [ ]

**Why:** Static loss weighting scales gradients but does not change how often each dataset
is seen. Sampling proportional to recent loss improvement rate ensures a stuck dataset
gets more training signal automatically.

- [ ] Track EMA of per-dataset loss delta across steps (suggested alpha=0.1)
- [ ] Compute sampling probability as softmax over negative improvement rates
      (i.e. sample more from whichever dataset is improving least)
- [ ] Expose EMA alpha and initial weights as config parameters
- [ ] Log sampling weights per epoch to monitor balance over training

```python
# Reference sketch
ema_delta = {ds: 0.0 for ds in datasets}
alpha = 0.1

def update_sampling_weights(current_losses, prev_losses):
    for ds in datasets:
        delta = prev_losses[ds] - current_losses[ds]  # positive = improving
        ema_delta[ds] = alpha * delta + (1 - alpha) * ema_delta[ds]
    weights = softmax([-ema_delta[ds] for ds in datasets])
    return weights
```

---

## Notes

- Items 1–3 are the highest-leverage changes and should be done before tuning sampling.
- Loss reweighting alone (already attempted) is essentially item 5 without items 1–4 in
  place, which is why it has not helped.
- Track US-Nerve and IDRiD metrics separately in all experiments — joint metrics will
  mask IDRiD's plateau.
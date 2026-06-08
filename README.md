# ternary-optimizer

Sign-based optimization for networks with ternary weights {-1, 0, +1}.

## The Problem

Standard gradient descent computes a continuous update for each parameter: "move this weight by 0.0037." But if your weight can only be -1, 0, or +1, the gradient's magnitude is irrelevant — only its direction matters. The gradient says "push toward +1" or "push toward -1." The size of the push is noise.

This isn't just a constraint of ternary networks. SignSGD (Bernstein et al., 2018) showed that replacing gradients with their signs can match or exceed standard Adam on deep networks. The sign carries the direction; the learning rate carries the magnitude. Everything else is variance that averages out over a batch.

The practical payoff: sign-based updates use 2 bits per gradient instead of 32 bits. In distributed training, that's a 16× reduction in gradient communication. On constrained hardware, the update step becomes comparison and addition — no floating-point multiply-accumulate needed.

## The Insight

Three optimizer designs, all built on the same primitive:

1. **Ternary GD**: `θ -= lr × sign(∇L)`. The sign of the gradient, directly. Every parameter moves by exactly ±lr. Simple, correct on average for smooth convex functions.

2. **Ternary Momentum**: `m = β·m + ∇L; θ -= lr × sign(m)`. The momentum buffer is full-precision, accumulating gradient history. Only the *sign of the momentum* drives the update. This is the key design choice: momentum is accumulated normally (so it reflects true gradient direction), but the step is ternary (so it's communication-efficient and hardware-friendly). Even if individual gradient signs flip-flop, the accumulated momentum maintains a consistent direction.

3. **Ternary Adam**: Full Adam structure — first moment `m`, second moment `v`, bias correction — but uses `sign(m̂)` for the step instead of `m̂ / (√v̂ + ε)`. The second moment is tracked but doesn't scale the update. This is SignSGD with Adam's memory structure, not Adam with ternary outputs.

After training in full precision, you ternarize the weights for deployment. The optimal scale factor `s = Σ(wᵢ·Tᵢ) / Σ(Tᵢ²)` minimizes `‖W - s·T‖²`, recovering most of the accuracy gap.

## How It Works

### Weight ternarization

Two strategies for choosing the threshold:

- **Fixed**: a constant τ. Weights with `|w| > τ` become ±1, the rest become 0.
- **MaxScaled**: `τ = α × max(|w|)`. The threshold adapts to the weight distribution — if the largest weight is 10.0 and α = 0.05, τ = 0.5. This preserves the shape of the distribution.

After thresholding, the optimal scale is computed by least-squares projection of the original weights onto the ternary weights. During inference: `output ≈ scale × (T × input)`.

### Learning rate schedule

A ternary-aware schedule: increase when loss improves, decrease after `patience` consecutive non-improving steps. The adjustment is discrete (increase/stay/decrease), matching the discrete nature of the parameter space. Bounded by `min_lr` and `max_lr`.

### The convergence behavior

Sign-based optimizers don't converge to a point — they oscillate around the optimum with amplitude `lr`. The test suite verifies that `TernaryGD` on `f(x) = x²` drives `|x|` below 1.0 within 200 steps with lr=0.1. The residual oscillation is inherent to the method; you'd need a diminishing learning rate schedule for tighter convergence.

## Code Example

```rust
use ternary_optimizer::*;

// ── Ternary Gradient Descent ──
let opt = TernaryGD::new(0.1);
let mut params = vec![5.0, -3.0, 0.5];
let grad = vec![2.0, -1.0, 0.001];
opt.step(&mut params, &grad);
// params[0] = 5.0 - 0.1*sign(2.0) = 4.9
// params[1] = -3.0 - 0.1*sign(-1.0) = -2.9
// params[2] = 0.5 - 0.1*sign(0.001) = 0.4

// ── Ternary Momentum ──
let mut opt = TernaryMomentum::new(3, 0.5, 0.9);
let mut params = vec![10.0, 5.0, -3.0];
for _ in 0..10 {
    let grad = vec![2.0 * params[0], 2.0 * params[1], 2.0 * params[2]];
    opt.step(&mut params, &grad);
}
// Momentum accumulates, smooths out gradient noise.
// Only sign(momentum) drives the step.

// ── Ternary Adam (SignAdam) ──
let mut opt = TernaryAdam::new(3, 0.01).with_betas(0.9, 0.999);
let mut params = vec![10.0, 5.0, -3.0];
for _ in 0..100 {
    let grad = vec![2.0 * params[0], 2.0 * params[1], 2.0 * params[2]];
    opt.step(&mut params, &grad);
}
// Bias-corrected first moment → sign() → step

// ── Weight ternarization ──
let weights = vec![0.8, -0.9, 0.1, -0.05, 0.95, -0.3];
let (ternary, scale) = ternarize_weights(&weights, &TernarizeStrategy::Fixed(0.5));
// ternary: [1, -1, 0, 0, 1, 0], scale ≈ 0.87

let (ternary2, scale2) = ternarize_weights(
    &weights,
    &TernarizeStrategy::MaxScaled { alpha: 0.1 },
);
// threshold = 0.1 * max(|w|) = 0.095

// ── Learning rate schedule ──
let mut sched = TernaryLRSchedule::new(0.01, 0.0001, 0.1);
let (adj, lr) = sched.step(0.5);    // loss improved → Increase
let (adj, lr) = sched.step(0.6);    // no improvement → Stay
// ...after `patience` bad steps → Decrease

// ── Utility: compute ternary gradient ──
let grad = vec![-3.0, 0.0, 5.0, -0.001, 100.0];
let tgrad = ternary_gradient(&grad);
// [-1.0, 0.0, 1.0, -1.0, 1.0]
```

## Module Map

Everything in `src/lib.rs`.

```
TernaryParams          — wrapper around Vec<f64> with .ternarize(threshold)

sign(v)                — extract sign: -1.0, 0.0, or +1.0
ternarize_value(v, τ)  — single value to {-1, 0, +1}
ternary_gradient(g)    — element-wise sign of a gradient vector

TernaryGD              — sign(gradient) optimizer
  .step(params, grad)  — in-place update

TernaryMomentum        — sign(momentum) optimizer
  .new(size, lr, β)
  .step(params, grad)  — updates momentum buffer, then sign step

TernaryAdam            — sign(bias-corrected m) optimizer
  .new(size, lr)
  .with_betas(β₁, β₂) — builder pattern
  .step(params, grad)  — updates m, v; bias-corrects; sign step

TernarizeStrategy      — enum { Fixed(τ), MaxScaled{α} }
ternarize_weights(w, s) — → (ternary_vec, optimal_scale)
ternarization_sparsity(t) — fraction of zeros

TernaryLRSchedule      — discrete increase/stay/decrease scheduler
  .new(init, min, max)
  .step(loss)          — → (LRAdjustment, new_lr)
```

## Design Decisions

**The second moment is tracked but unused.** `TernaryAdam` maintains `v` (the uncentered variance estimate) identically to standard Adam, but the update is `lr × sign(m̂)`, not `lr × m̂ / (√v̂ + ε)`. The second moment is there for diagnostics — you can inspect `opt.v` to understand gradient variance — but it doesn't affect the step. This is intentional: the whole point is to use the sign only. If you want the second moment to scale the step, use a different optimizer.

**f64 for parameters, not f32.** The training loop operates on `f64`. The `ternary-quantize` crate uses `f32`. This precision boundary exists because optimization accumulates floating-point error over many steps (bias correction, momentum decay), where f64's extra mantissa bits matter. The final ternarization step erases all precision anyway.

**No parameter groups.** PyTorch-style optimizers let you assign different learning rates to different parameter groups. This crate applies the same lr to all parameters. For ternary networks, this is usually fine — the sign-based update normalizes away scale differences — but it limits flexibility for layer-wise learning rate schedules.

**The learning rate schedule is loss-based, not epoch-based.** It adjusts based on the training loss, not the step count. There's no warmup phase and no cosine annealing. These could be added, but the ternary oscillation behavior makes smooth schedules less meaningful — you're already in a discrete regime.

**TernaryParams is a thin wrapper.** It holds `Vec<f64>` and provides `.ternarize()`. It doesn't enforce that parameters stay in {-1, 0, +1} during training — they're full-precision throughout, and ternarization happens at the end. This is the standard approach (train full-precision, quantize for deployment), not the alternative (force ternary weights at every step).

## Status

- **15 tests passing.** Basic descent convergence on x², sign update correctness, momentum accumulation and direction persistence through gradient sign changes, SignAdam matching manual calculation, bias correction verification, weight ternarization distribution preservation, scale factor correctness, MaxScaled threshold strategy, sparsity computation, LR schedule increase/decrease/bounds.
- **Functional for research and small-scale training.** The optimizers are correct and converge on simple objectives.
- **Known gaps:**
  - No parameter groups (can't set per-layer learning rates)
  - No gradient clipping or warmup
  - No distributed gradient aggregation (the sign-based update is communication-efficient, but the infrastructure isn't implemented)
  - TernaryAdam's second moment is computed but doesn't affect the step
  - No convergence guarantee for non-convex objectives
  - Oscillates around the optimum with amplitude ≈ lr (inherent to sign-based methods)

## Ecosystem

- [`ternary-quantize`](https://github.com/SuperInstance/ternary-quantize) — post-training quantization (the f32→ternary step)
- [`ternary-svm`](https://github.com/SuperInstance/ternary-svm) — classification on ternary features
- [`ternary-em`](https://github.com/SuperInstance/ternary-em) — mixture modeling for ternary distributions
- [`ternary-types`](https://github.com/SuperInstance/ternary-types) — shared trait definitions

## References

- Bernstein, J., Wang, Y.-X., Azizzadenesheli, K., & Anandkumar, A. (2018). *signSGD: Compressed Optimisation for Non-Convex Problems*. [arXiv:1802.04434](https://arxiv.org/abs/1802.04434)
- Li, F. et al. (2016). *Ternary Weight Networks*. [arXiv:1605.04711](https://arxiv.org/abs/1605.04711)

## License

MIT

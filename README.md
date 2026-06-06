# ternary-optimizer

**Ternary optimization algorithms for neural networks with {-1, 0, +1} weights.**

[![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)]()

Training ternary neural networks — where weights are constrained to {-1, 0, +1} — requires
specialized optimization algorithms. Standard gradient descent produces continuous updates that
must be quantized, introducing noise and instability. **ternary-optimizer** provides sign-based
and ternary-constrained optimizers that operate directly in ternary space, inspired by
SignSGD, ternary weight networks, and ultra-low-precision training research.

## Why Ternary Optimization?

In a ternary network, gradients carry more information than needed. The sign of the gradient
tells you the direction to move; the magnitude is less useful when your parameters can only
take three values. Sign-based optimization:

1. **Reduces communication** — only 2 bits per gradient element (sign + zero)
2. **Improves robustness** — sign is invariant to gradient scaling
3. **Enables integer-only training** — no floating-point multiply-accumulate needed
4. **Compresses updates** — ternary gradients are 16× smaller than FP32

## Features

- **Ternary Gradient Descent** — sign-only updates: `θ ← θ - lr × sign(∇L)`
- **Ternary Momentum** — accumulates direction over time, uses sign of momentum buffer
- **Ternary Adam** — Adam-style moment tracking with sign-based updates (SignSGD + Adam structure)
- **Weight Ternarization** — convert full-precision weights to {-1, 0, +1} with optimal scaling
- **Ternary Learning Rate Schedule** — discrete LR adjustment: increase / stay / decrease

## Quick Start

```rust
use ternary_optimizer::*;

// ── Ternary Gradient Descent ──
let mut params = vec![5.0, -3.0, 0.5];
let grad = vec![2.0, -1.0, 0.001]; // gradient of the loss
let opt = TernaryGD::new(0.1);
opt.step(&mut params, &grad);
// params updated by -lr * sign(grad)

// ── Ternary Momentum ──
let mut opt = TernaryMomentum::new(3, 0.5, 0.9);
let mut params = vec![10.0, 5.0, -3.0];
for _ in 0..10 {
    let grad = vec![2.0 * params[0], 2.0 * params[1], 2.0 * params[2]];
    opt.step(&mut params, &grad);
}

// ── Ternary Adam ──
let mut opt = TernaryAdam::new(3, 0.01).with_betas(0.9, 0.999);
let mut params = vec![10.0, 5.0, -3.0];
for _ in 0..100 {
    let grad = vec![2.0 * params[0], 2.0 * params[1], 2.0 * params[2]];
    opt.step(&mut params, &grad);
}

// ── Weight Ternarization ──
let weights = vec![0.8, -0.9, 0.1, -0.05, 0.95, -0.3];
let (ternary, scale) = ternarize_weights(&weights, &TernarizeStrategy::Fixed(0.5));
// ternary = [1, -1, 0, 0, 1, 0], scale ≈ 0.87

// Ternarize relative to max magnitude
let (ternary2, scale2) = ternarize_weights(
    &weights,
    &TernarizeStrategy::MaxScaled { alpha: 0.1 },
);

// ── Learning Rate Schedule ──
let mut sched = TernaryLRSchedule::new(0.01, 0.0001, 0.1);
let (adj, lr) = sched.step(0.5); // loss improved → Increase
assert_eq!(adj, LRAdjustment::Increase);
```

## Algorithm Details

### Ternary Gradient Descent (TernaryGD)

The simplest ternary optimizer. Uses only the sign of each gradient component:

```
θ_i ← θ_i - lr × sign(∇_i L(θ))
```

Each parameter moves by exactly `±lr` or stays put. Convergence relies on the sign
being correct on average, which holds for smooth convex functions.

### Ternary Momentum

Accumulates a full-precision momentum buffer, then uses the sign of the momentum
for the update:

```
m ← β × m + ∇L(θ)
θ ← θ - lr × sign(m)
```

The momentum buffer smooths out gradient noise. Even if individual gradient signs
flip, the accumulated momentum maintains a consistent direction.

### Ternary Adam (SignAdam)

Combines Adam's moment tracking with sign-based updates:

```
m ← β₁ × m + (1 - β₁) × g           # first moment
v ← β₂ × v + (1 - β₂) × g²          # second moment (diagnostic only)
m̂ ← m / (1 - β₁ᵗ)                    # bias correction
θ ← θ - lr × sign(m̂)                 # sign-based update
```

The second moment `v` is tracked for potential use in adaptive methods but does not
scale the step size. This is inspired by **SignSGD** (Bernstein et al., 2018) which
showed that sign-based updates can match or exceed Adam's performance on deep networks.

### Weight Ternarization

Converts full-precision weights to {-1, 0, +1} and computes an optimal scale factor:

```
threshold = α × max(|w|)    (MaxScaled strategy)
T_i = sign(w_i)  if |w_i| > threshold, else 0
scale = Σ(w_i × T_i) / Σ(T_i²)
```

The scale factor minimizes `||W - s × T||²`, allowing approximate recovery of the
original weights during inference: `output ≈ scale × (T × input)`.

### Ternary Learning Rate Schedule

Adjusts the learning rate in discrete steps based on loss trend:

- **Improve** (loss < best) → optionally increase LR by `increase_factor`
- **Plateau** (loss ≥ best, within patience) → LR stays
- **Stagnate** (patience exceeded) → decrease LR by `decrease_factor`

Clamped to `[min_lr, max_lr]` bounds.

## Research Context

- **SignSGD**: Bernstein et al. "signSGD: Compressed optimisation for non-convex problems."
  *ICML* (2018).
- **Ternary Weight Networks**: Li et al. "Ternary weight networks." *NIPS Workshop* (2016).
- **Trained Ternary Quantization**: Zhu et al. "Trained ternary quantization." *ICLR* (2017).
- **1-bit SGD**: Seide et al. "1-bit stochastic gradient descent and its application to
  data-parallel distributed training of speech DNNs." *Interspeech* (2014).

## Testing

```bash
cargo test
```

15 comprehensive tests covering:
- Basic descent convergence on x²
- Sign update correctness
- Momentum accumulation and direction persistence
- SignAdam matching manual calculation
- Bias correction verification
- Weight ternarization distribution preservation
- Scale factor correctness
- Max-scaled threshold strategy
- Sparsity computation
- Learning rate schedule increase/decrease/bounds

## License

MIT

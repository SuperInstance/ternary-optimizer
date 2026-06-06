# ternary-optimizer

**Training ternary networks with sign-based updates — because the gradient's direction matters more than its magnitude.**

[![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)]()
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Why This Exists

Training a neural network with ternary weights {-1, 0, +1} is inherently strange. Standard gradient descent produces continuous updates: "move this parameter by 0.0037." But your parameter can only be -1, 0, or +1. The gradient's magnitude is useless — only its sign matters.

This isn't a limitation. It's an insight. **SignSGD** (Bernstein et al., 2018) showed that sign-based updates can match or exceed Adam on deep networks. The sign carries the direction; the learning rate carries the step size. Everything else is noise.

This crate implements optimizers that embrace this constraint: updates based on the sign of the gradient (or momentum, or bias-corrected first moment), with weight ternarization as a first-class operation.

## The Key Insight

In a ternary network, the gradient tells you which *direction* to push a weight. If a weight is currently 0, the gradient says "move toward +1" or "move toward -1." The magnitude of the gradient is irrelevant — your destination is one of three values. This reduces communication by 16× (2 bits per gradient instead of 32) and eliminates the need for floating-point multiply-accumulate during the update.

## Quick Start

```toml
[dependencies]
ternary-optimizer = "0.1"
```

```rust
use ternary_optimizer::*;

// ── Ternary Gradient Descent ──
let mut params = vec![5.0, -3.0, 0.5];
let grad = vec![2.0, -1.0, 0.001];
let opt = TernaryGD::new(0.1);
opt.step(&mut params, &grad);
// Each param moves by ±lr (sign of gradient)

// ── Ternary Momentum ──
let mut opt = TernaryMomentum::new(3, 0.5, 0.9);
let mut params = vec![10.0, 5.0, -3.0];
for _ in 0..10 {
    let grad = vec![2.0 * params[0], 2.0 * params[1], 2.0 * params[2]];
    opt.step(&mut params, &grad);
}

// ── Ternary Adam (SignAdam) ──
let mut opt = TernaryAdam::new(3, 0.01).with_betas(0.9, 0.999);
let mut params = vec![10.0, 5.0, -3.0];
for _ in 0..100 {
    let grad = vec![2.0 * params[0], 2.0 * params[1], 2.0 * params[2]];
    opt.step(&mut params, &grad);
}

// ── Weight Ternarization ──
let weights = vec![0.8, -0.9, 0.1, -0.05, 0.95, -0.3];
let (ternary, scale) = ternarize_weights(&weights, &TernarizeStrategy::Fixed(0.5));
// ternary: [1, -1, 0, 0, 1, 0], scale ≈ 0.87

// Max-scaled: adapt threshold to weight distribution
let (t2, s2) = ternarize_weights(&weights, &TernarizeStrategy::MaxScaled { alpha: 0.1 });

// ── Learning Rate Schedule ──
let mut sched = TernaryLRSchedule::new(0.01, 0.0001, 0.1);
let (adj, lr) = sched.step(0.5); // loss improved → increase LR
```

## Architecture

```
                    ┌─────────────┐
                    │  Gradient   │  ∇L(θ)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌──▼──────────┐ ┌▼──────────────┐
       │  TernaryGD  │ │  TernaryMom │ │  TernaryAdam  │
       │ sign(∇)     │ │ sign(βm+∇) │ │ sign(m̂_bc)   │
       └──────┬──────┘ └──┬──────────┘ └──┬────────────┘
              │            │               │
              └────────────┼───────────────┘
                           │
                    ┌──────▼──────┐
                    │  θ -= lr   │
                    │  × sign(.) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ ternarize_  │
                    │ weights()   │  (optional, post-training)
                    └─────────────┘
```

## Algorithm Details

### Ternary Gradient Descent

```
θ_i ← θ_i - lr × sign(∇_i L(θ))
```

The simplest sign-based optimizer. Every parameter moves by exactly `±lr` or stays put. Convergence relies on the sign being correct on average — which holds for smooth convex functions and, empirically, for deep networks.

### Ternary Momentum

```
m ← β × m + ∇L(θ)
θ ← θ - lr × sign(m)
```

The momentum buffer is full-precision, accumulating gradient history. Only the *sign of the momentum* drives the update. This smooths out oscillation: even if individual gradient signs flip, the accumulated momentum maintains a consistent direction.

### Ternary Adam (SignAdam)

```
m ← β₁ × m + (1 - β₁) × g         # first moment (bias-corrected)
v ← β₂ × v + (1 - β₂) × g²        # second moment (tracked, not used for scaling)
m̂ = m / (1 - β₁ᵗ)                  # bias correction
θ ← θ - lr × sign(m̂)               # sign-based update
```

Adam's moment tracking with sign-based updates. The second moment `v` is tracked for diagnostics but doesn't scale the step — unlike standard Adam where `v̂` appears in the denominator. This is SignSGD with Adam's memory structure.

### Weight Ternarization

After training (full-precision), convert weights to {-1, 0, +1} with an optimal scale factor:

```
threshold = α × max(|w|)    (MaxScaled strategy)
T_i = sign(w_i) if |w_i| > threshold, else 0
scale = Σ(w_i × T_i) / Σ(T_i²)
```

The scale factor minimizes `||W - s × T||²`. During inference: `output ≈ scale × (T × input)`. You keep one float per layer and recover most of the lost accuracy.

## API Reference

### Optimizers

```rust
struct TernaryGD { pub learning_rate: f64 }
impl TernaryGD {
    fn new(learning_rate: f64) -> Self;
    fn step(&self, params: &mut [f64], grad: &[f64]);
}

struct TernaryMomentum { pub learning_rate: f64, pub beta: f64, pub momentum: Vec<f64> }
impl TernaryMomentum {
    fn new(param_size: usize, learning_rate: f64, beta: f64) -> Self;
    fn step(&mut self, params: &mut [f64], grad: &[f64]);
}

struct TernaryAdam { pub learning_rate: f64, pub beta1: f64, pub beta2: f64, pub t: u64, ... }
impl TernaryAdam {
    fn new(param_size: usize, learning_rate: f64) -> Self;
    fn with_betas(self, beta1: f64, beta2: f64) -> Self;
    fn step(&mut self, params: &mut [f64], grad: &[f64]);
}
```

### Weight Ternarization

```rust
enum TernarizeStrategy {
    Fixed(f64),
    MaxScaled { alpha: f64 },
}

fn ternarize_weights(weights: &[f64], strategy: &TernarizeStrategy) -> (Vec<f64>, f64);
fn ternarization_sparsity(ternary: &[f64]) -> f64;
```

### Learning Rate Schedule

```rust
enum LRAdjustment { Increase, Stay, Decrease }

struct TernaryLRSchedule { pub lr: f64, pub min_lr: f64, pub max_lr: f64, ... }
impl TernaryLRSchedule {
    fn new(initial_lr: f64, min_lr: f64, max_lr: f64) -> Self;
    fn step(&mut self, loss: f64) -> (LRAdjustment, f64);
    fn current_lr(&self) -> f64;
}
```

### Utilities

```rust
fn sign(v: f64) -> f64;
fn ternarize_value(v: f64, threshold: f64) -> f64;
fn ternary_gradient(grad: &[f64]) -> Vec<f64>;
```

## Real-World Example: Training a Ternary Classifier on the Edge

You have a dataset of 10,000 labeled examples and need to train a ternary classifier that will run on a microcontroller. The model is small (1000 parameters), but training on-device saves a cloud round-trip.

```rust
let mut opt = TernaryAdam::new(1000, 0.01).with_betas(0.9, 0.999);
let mut lr_sched = TernaryLRSchedule::new(0.01, 0.001, 0.1);
let mut params = vec![0.0; 1000];

for epoch in 0..50 {
    let mut total_loss = 0.0;
    for (batch_x, batch_y) in dataloader {
        let grad = compute_gradient(&params, batch_x, batch_y);
        opt.step(&mut params, &grad);
        total_loss += grad.iter().map(|g| g.abs()).sum::<f64>();
    }
    let (adj, lr) = lr_sched.step(total_loss);
}

// Ternarize for deployment
let (ternary_weights, scale) = ternarize_weights(&params, &TernarizeStrategy::MaxScaled { alpha: 0.05 });
// Deploy ternary_weights + scale to microcontroller
```

Sign-based updates use no floating-point multiply during the weight update — just comparison and addition. The entire training loop can run in integer arithmetic on a constrained device.

## Performance Characteristics

- **TernaryGD**: O(n) per step — one sign computation per parameter
- **TernaryMomentum**: O(n) per step — one multiply-add + one sign per parameter
- **TernaryAdam**: O(n) per step — two multiply-adds + bias correction + one sign per parameter
- **Weight ternarization**: O(n) — one pass for thresholding, one pass for scale computation

Memory: TernaryAdam uses 2n extra storage (first + second moment). TernaryMomentum uses n. TernaryGD uses none.

The learning rate schedule uses O(1) memory and O(1) computation per step.

## Ecosystem Connections

The optimizer is the training loop of the ternary stack:

- [`ternary-loss`](https://github.com/SuperInstance/ternary-loss) — computes the gradients this optimizer consumes
- [`ternary-norm`](https://github.com/SuperInstance/ternary-norm) — γ and β parameters are updated by this optimizer
- [`ternary-activation`](https://github.com/SuperInstance/ternary-activation) — straight-through estimator bridges ternary activations with sign-based gradients
- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — the core operation being optimized

## Open Questions

- **Ternary-aware LR scheduling**: The current schedule is loss-based. A schedule aware of the ternary quantization error (distance between full-precision and ternary weights) could be more principled.
- **Gradient quantization**: Currently full-precision gradients are reduced to signs. Could we extract more information with 2-bit gradient quantization while staying ternary-friendly?
- **Distributed sign-SGD**: Sign-based gradients are naturally suited for distributed training (1-bit communication). The infrastructure isn't here yet.

## Testing

```bash
cargo test
```

15 tests covering: basic descent convergence on x², sign update correctness, momentum accumulation and direction persistence, SignAdam matching manual calculation, bias correction verification, weight ternarization distribution preservation, scale factor correctness, MaxScaled threshold strategy, sparsity computation, learning rate schedule increase/decrease/bounds.

## License

MIT

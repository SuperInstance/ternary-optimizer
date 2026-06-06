//! # ternary-optimizer
//!
//! Ternary optimization algorithms for neural networks with {-1, 0, +1} weights.
//!
//! This crate implements sign-based and ternary-constrained optimization methods
//! inspired by SignSGD, ternary weight networks, and ultra-low-precision training.
//! Gradients are reduced to their sign, and updates are ternary-valued.


/// A parameter vector with full-precision storage that can be ternarized.
#[derive(Clone, Debug)]
pub struct TernaryParams {
    pub data: Vec<f64>,
}

impl TernaryParams {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Ternarize parameters to {-1, 0, +1} with the given threshold.
    pub fn ternarize(&self, threshold: f64) -> Vec<f64> {
        self.data.iter().map(|&v| ternarize_value(v, threshold)).collect()
    }
}

/// Extract the sign of a gradient value: -1.0, 0.0, or +1.0.
pub fn sign(v: f64) -> f64 {
    if v > 0.0 { 1.0 }
    else if v < 0.0 { -1.0 }
    else { 0.0 }
}

/// Ternarize a single value to {-1, 0, +1} based on threshold.
pub fn ternarize_value(v: f64, threshold: f64) -> f64 {
    if v > threshold { 1.0 }
    else if v < -threshold { -1.0 }
    else { 0.0 }
}

/// Compute ternary gradient: element-wise sign of the gradient.
pub fn ternary_gradient(grad: &[f64]) -> Vec<f64> {
    grad.iter().map(|&v| sign(v)).collect()
}

// ── Ternary Gradient Descent ─────────────────────────────────────────────────

/// Ternary Gradient Descent optimizer.
///
/// Updates parameters using only the sign of the gradient:
/// `θ ← θ - lr × sign(∇L(θ))`
pub struct TernaryGD {
    pub learning_rate: f64,
}

impl TernaryGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }

    /// Perform a single update step.
    pub fn step(&self, params: &mut [f64], grad: &[f64]) {
        assert_eq!(params.len(), grad.len());
        for (p, &g) in params.iter_mut().zip(grad.iter()) {
            *p -= self.learning_rate * sign(g);
        }
    }
}

// ── Ternary Momentum ─────────────────────────────────────────────────────────

/// Ternary Momentum optimizer.
///
/// Accumulates a momentum buffer in full precision, but uses the sign of the
/// momentum for the actual update:
/// ```text
/// m = beta * m + grad
/// theta = theta - lr * sign(m)
/// ```
pub struct TernaryMomentum {
    pub learning_rate: f64,
    pub beta: f64,
    pub momentum: Vec<f64>,
}

impl TernaryMomentum {
    pub fn new(param_size: usize, learning_rate: f64, beta: f64) -> Self {
        Self {
            learning_rate,
            beta,
            momentum: vec![0.0; param_size],
        }
    }

    /// Perform a single update step.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        assert_eq!(params.len(), grad.len());
        assert_eq!(params.len(), self.momentum.len());

        for i in 0..params.len() {
            self.momentum[i] = self.beta * self.momentum[i] + grad[i];
            params[i] -= self.learning_rate * sign(self.momentum[i]);
        }
    }
}

// ── Ternary Adam (Sign-based) ────────────────────────────────────────────────

/// Ternary Adam optimizer (sign-based, inspired by SignSGD + Adam structure).
///
/// Maintains first and second moment estimates like Adam, but uses only the
/// sign of the bias-corrected first moment for updates:
/// ```text
/// m = beta1 * m + (1 - beta1) * g
/// v = beta2 * v + (1 - beta2) * g^2
/// m_hat = m / (1 - beta1^t)
/// theta = theta - lr * sign(m_hat)
/// ```
/// The second moment `v` is tracked for diagnostic purposes but does not
/// scale the step size (unlike standard Adam).
pub struct TernaryAdam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub m: Vec<f64>,  // first moment
    pub v: Vec<f64>,  // second moment
    pub t: u64,       // timestep
}

impl TernaryAdam {
    pub fn new(param_size: usize, learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: vec![0.0; param_size],
            v: vec![0.0; param_size],
            t: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Perform a single update step.
    pub fn step(&mut self, params: &mut [f64], grad: &[f64]) {
        assert_eq!(params.len(), grad.len());
        assert_eq!(params.len(), self.m.len());
        self.t += 1;

        let bc1 = 1.0 - self.beta1.powi(self.t as i32); // bias correction
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bc1;
            let _v_hat = self.v[i] / bc2; // tracked but not used for scaling
            params[i] -= self.learning_rate * sign(m_hat);
        }
    }
}

// ── Weight Ternarization ─────────────────────────────────────────────────────

/// Strategy for choosing the ternarization threshold.
#[derive(Clone, Debug)]
pub enum TernarizeStrategy {
    /// Fixed threshold value.
    Fixed(f64),
    /// Threshold = α × max(|w|), where α is typically 0.05 to 0.1.
    /// This preserves the distribution of weights by scaling relative to the max.
    MaxScaled { alpha: f64 },
}

/// Ternarize full-precision weights to {-1, 0, +1}.
///
/// Returns (ternarized_weights, scale_factor) where the scale factor can be
/// used during inference: `output ≈ scale × (ternary_weights × input)`.
pub fn ternarize_weights(weights: &[f64], strategy: &TernarizeStrategy) -> (Vec<f64>, f64) {
    let threshold = match strategy {
        TernarizeStrategy::Fixed(t) => *t,
        TernarizeStrategy::MaxScaled { alpha } => {
            let max_abs = weights.iter().map(|w| w.abs()).fold(0.0_f64, f64::max);
            alpha * max_abs
        }
    };

    let ternary: Vec<f64> = weights.iter().map(|&w| ternarize_value(w, threshold)).collect();

    // Compute optimal scale factor: minimize ||W - s * T||²
    // s = sum(W_i * T_i) / sum(T_i * T_i)
    let numerator: f64 = weights.iter().zip(ternary.iter())
        .map(|(w, t)| w * t).sum();
    let denominator: f64 = ternary.iter().map(|t| t * t).sum();
    let scale = if denominator > 0.0 { numerator / denominator } else { 0.0 };

    (ternary, scale)
}

/// Compute the fraction of weights that are preserved (non-zero) after ternarization.
pub fn ternarization_sparsity(ternary: &[f64]) -> f64 {
    let zeros = ternary.iter().filter(|&&v| v == 0.0).count();
    zeros as f64 / ternary.len() as f64
}

// ── Learning Rate Schedule ───────────────────────────────────────────────────

/// Ternary learning rate adjustment direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LRAdjustment {
    Increase,
    Stay,
    Decrease,
}

/// A simple ternary learning rate scheduler.
///
/// Adjusts the learning rate in discrete steps: increase, stay, or decrease.
/// The decision is based on the loss trend over recent steps.
pub struct TernaryLRSchedule {
    pub lr: f64,
    pub min_lr: f64,
    pub max_lr: f64,
    pub increase_factor: f64,
    pub decrease_factor: f64,
    pub patience: usize,
    pub best_loss: f64,
    pub bad_steps: usize,
}

impl TernaryLRSchedule {
    pub fn new(initial_lr: f64, min_lr: f64, max_lr: f64) -> Self {
        Self {
            lr: initial_lr,
            min_lr,
            max_lr,
            increase_factor: 1.2,
            decrease_factor: 0.5,
            patience: 5,
            best_loss: f64::INFINITY,
            bad_steps: 0,
        }
    }

    /// Get the current learning rate.
    pub fn current_lr(&self) -> f64 {
        self.lr
    }

    /// Determine adjustment based on the current loss, then apply it.
    /// Returns the adjustment direction and the new learning rate.
    pub fn step(&mut self, loss: f64) -> (LRAdjustment, f64) {
        if loss < self.best_loss {
            self.best_loss = loss;
            self.bad_steps = 0;
            // Improvement: optionally increase LR
            let new_lr = (self.lr * self.increase_factor).min(self.max_lr);
            if new_lr > self.lr {
                self.lr = new_lr;
                return (LRAdjustment::Increase, self.lr);
            }
            return (LRAdjustment::Stay, self.lr);
        } else {
            self.bad_steps += 1;
            if self.bad_steps >= self.patience {
                let new_lr = (self.lr * self.decrease_factor).max(self.min_lr);
                self.bad_steps = 0;
                if new_lr < self.lr {
                    self.lr = new_lr;
                    return (LRAdjustment::Decrease, self.lr);
                }
                return (LRAdjustment::Stay, self.lr);
            }
            return (LRAdjustment::Stay, self.lr);
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_descent_converges() {
        // Minimize f(x) = x² using ternary gradient descent
        let mut params = TernaryParams::new(vec![10.0]);
        let opt = TernaryGD::new(0.1);

        for _ in 0..200 {
            // grad of x² = 2x
            let grad = vec![2.0 * params.data[0]];
            opt.step(&mut params.data, &grad);
        }

        // Should converge near 0 (may oscillate around it)
        assert!(params.data[0].abs() < 1.0, "should converge, got {}", params.data[0]);
    }

    #[test]
    fn test_ternary_gd_sign_update() {
        let mut params = vec![5.0];
        let grad = vec![3.0]; // positive gradient → decrease param
        let opt = TernaryGD::new(1.0);
        opt.step(&mut params, &grad);
        assert_eq!(params[0], 4.0); // 5 - 1*sign(3) = 5 - 1 = 4
    }

    #[test]
    fn test_momentum_accumulates_direction() {
        let mut opt = TernaryMomentum::new(1, 1.0, 0.9);
        let mut params = vec![10.0];

        // Repeated positive gradients → momentum builds up
        for _ in 0..5 {
            let grad = vec![2.0];
            opt.step(&mut params, &grad);
        }

        // After 5 steps of positive gradient, momentum should be accumulated
        // momentum after step 5: 0.9^4*0.9*2 + 0.9^3*2 + ... + 2
        // All updates are -1.0 * lr * sign(momentum) = -1.0 since momentum stays positive
        // param = 10 - 5 = 5
        assert!(params[0] < 10.0, "params should have decreased: {}", params[0]);
        assert!(opt.momentum[0] > 0.0, "momentum should be positive: {}", opt.momentum[0]);
    }

    #[test]
    fn test_momentum_direction_change() {
        let mut opt = TernaryMomentum::new(1, 1.0, 0.5);

        // Build positive momentum
        let mut params = vec![0.0];
        opt.step(&mut params, &vec![1.0]);
        assert_eq!(params[0], -1.0); // sign(0.5*0 + 1) = +1, so param decreases

        // Now negative gradient, but momentum still positive
        opt.step(&mut params, &vec![-0.3]);
        // momentum = 0.5*1.0 + (-0.3) = 0.2, still positive
        assert_eq!(params[0], -2.0); // still moving in the same direction
    }

    #[test]
    fn test_sign_adam_matches_manual() {
        let mut opt = TernaryAdam::new(1, 1.0).with_betas(0.0, 0.0); // β=0 means no accumulation
        let mut params = vec![5.0];

        // With β1=0: m = (1-0)*grad = grad; m_hat = grad/1 = grad
        opt.step(&mut params, &vec![3.0]);
        assert_eq!(params[0], 4.0); // 5 - 1*sign(3) = 4
    }

    #[test]
    fn test_sign_adam_bias_correction() {
        let mut opt = TernaryAdam::new(1, 1.0).with_betas(0.9, 0.999);
        let mut params = vec![5.0];

        // Step 1: m = 0.9*0 + 0.1*3 = 0.3, m_hat = 0.3/(1-0.9) = 3.0
        opt.step(&mut params, &vec![3.0]);
        assert_eq!(params[0], 4.0); // 5 - sign(3.0) = 4

        // Verify internal state
        assert!((opt.m[0] - 0.3).abs() < 1e-10);
        assert_eq!(opt.t, 1);
    }

    #[test]
    fn test_ternarization_preserves_distribution() {
        // Create weights with known distribution: ~1/3 each of {-1, 0, +1} range
        let weights: Vec<f64> = (0..300).map(|i| {
            match i % 3 {
                0 => 0.9,   // should become +1
                1 => -0.9,  // should become -1
                _ => 0.1,   // should become 0 (with threshold 0.5)
            }
        }).collect();

        let (ternary, _scale) = ternarize_weights(&weights, &TernarizeStrategy::Fixed(0.5));

        // Count distribution
        let n_pos = ternary.iter().filter(|&&v| v == 1.0).count();
        let n_neg = ternary.iter().filter(|&&v| v == -1.0).count();
        let n_zero = ternary.iter().filter(|&&v| v == 0.0).count();

        assert_eq!(n_pos, 100);
        assert_eq!(n_neg, 100);
        assert_eq!(n_zero, 100);
    }

    #[test]
    fn test_ternarization_scale_factor() {
        let weights = vec![1.0, -1.0, 0.5, -0.5];
        let (ternary, scale) = ternarize_weights(&weights, &TernarizeStrategy::Fixed(0.3));

        // All should ternarize to ±1
        assert_eq!(ternary, vec![1.0, -1.0, 1.0, -1.0]);

        // Scale = sum(w*t) / sum(t*t) = (1+1+0.5+0.5)/(1+1+1+1) = 3/4
        assert!((scale - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_max_scaled_ternarization() {
        let weights = vec![10.0, -10.0, 1.0, -1.0, 0.0];
        let (ternary, _scale) = ternarize_weights(
            &weights,
            &TernarizeStrategy::MaxScaled { alpha: 0.1 },
        );
        // threshold = 0.1 * 10 = 1.0
        assert_eq!(ternary[0], 1.0);   // 10 > 1.0
        assert_eq!(ternary[1], -1.0);  // -10 < -1.0
        assert_eq!(ternary[2], 0.0);   // 1.0 == threshold → not strictly greater
        assert_eq!(ternary[3], 0.0);   // -1.0 == -threshold → not strictly less
        assert_eq!(ternary[4], 0.0);
    }

    #[test]
    fn test_sparsity() {
        let ternary = vec![1.0, -1.0, 0.0, 0.0, 1.0];
        assert!((ternarization_sparsity(&ternary) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_schedule_decrease() {
        let mut sched = TernaryLRSchedule::new(0.1, 0.001, 1.0);

        // Establish best loss
        let (adj, _) = sched.step(1.0);
        assert_eq!(adj, LRAdjustment::Increase); // improved → increase
        assert!((sched.lr - 0.12).abs() < 1e-10); // 0.1 * 1.2

        // Bad steps
        for _ in 0..4 {
            let (adj, _) = sched.step(2.0);
            assert_eq!(adj, LRAdjustment::Stay);
        }
        // 5th bad step triggers decrease
        let (adj, _) = sched.step(2.0);
        assert_eq!(adj, LRAdjustment::Decrease);
    }

    #[test]
    fn test_schedule_respects_bounds() {
        let mut sched = TernaryLRSchedule::new(0.1, 0.05, 0.2);

        // Large improvement → tries to increase but max_lr=0.2
        sched.step(0.01);
        assert!(sched.lr <= 0.2 + 1e-10);

        // Reset and test min bound
        let mut sched2 = TernaryLRSchedule::new(0.06, 0.05, 1.0);
        sched2.patience = 1;
        sched2.best_loss = 0.001; // set low best
        let (adj, _) = sched2.step(1.0); // bad step
        assert_eq!(adj, LRAdjustment::Decrease);
        assert!(sched2.lr >= 0.05 - 1e-10);
    }

    #[test]
    fn test_ternary_gradient() {
        let grad = vec![-3.0, 0.0, 5.0, -0.001, 100.0];
        let tgrad = ternary_gradient(&grad);
        assert_eq!(tgrad, vec![-1.0, 0.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn test_sign_function() {
        assert_eq!(sign(5.0), 1.0);
        assert_eq!(sign(-5.0), -1.0);
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(sign(0.001), 1.0);
        assert_eq!(sign(-0.001), -1.0);
    }

    #[test]
    fn test_ternary_adam_multi_step() {
        let mut opt = TernaryAdam::new(2, 0.5).with_betas(0.9, 0.999);
        let mut params = vec![10.0, -10.0];

        for _ in 0..100 {
            // grad of x² = 2x → converges toward 0
            let grad = vec![2.0 * params[0], 2.0 * params[1]];
            opt.step(&mut params, &grad);
        }

        assert!(params[0].abs() < 2.0, "param[0] should converge, got {}", params[0]);
        assert!(params[1].abs() < 2.0, "param[1] should converge, got {}", params[1]);
    }
}

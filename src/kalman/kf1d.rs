/// One-dimensional Kalman filter.
///
/// The current belief about the state is a normal distribution N(x_hat, variance).
#[derive(Debug, Clone, Copy)]
pub struct Kalman1D {
    /// Best estimate (center of the bell curve)
    pub x_hat: f64,
    /// Uncertainty of the belief (variance)
    pub variance: f64,
    /// State transition factor: x_next = a * x
    a: f64,
    /// Process noise variance (how wrong the physics model is per step)
    q: f64,
    /// Measurement noise variance (how wrong the sensor is)
    r: f64,
}

impl Kalman1D {
    pub fn new(x_hat: f64, variance: f64, a: f64, q: f64, r: f64) -> Self {
        Self { x_hat, variance, a, q, r }
    }

    /// Predict step: time passes, the belief moves and widens.
    pub fn predict(&mut self) {
        self.x_hat = self.a * self.x_hat;
        self.variance = self.a * self.a * self.variance + self.q;
    }

    /// Update step: blend the belief with measurement `y`. Returns the Kalman gain.
    pub fn update(&mut self, y: f64) -> f64 {
        let k = self.variance / (self.variance + self.r);
        self.x_hat += k * (y - self.x_hat);
        self.variance *= 1.0 - k;
        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_two_gps_readings() {
        // GPS A (variance 4) reads 50 as the belief, GPS B (variance 100) reads 60 as the measurement
        let mut kf = Kalman1D::new(50.0, 4.0, 1.0, 0.0, 100.0);
        kf.update(60.0);
        assert!((kf.x_hat - 50.3846).abs() < 1e-3);
        assert!((kf.variance - 3.8462).abs() < 1e-3);
    }

    #[test]
    fn predict_widens_uncertainty() {
        let mut kf = Kalman1D::new(10.0, 4.0, 1.0, 1.0, 9.0);
        kf.predict();
        assert_eq!(kf.x_hat, 10.0);
        assert_eq!(kf.variance, 5.0);
    }
}
//! A car is parked at meter 100. The GPS is noisy with sigma = 3 meters.
//! Initially we know nothing: guess meter 0 with a huge uncertainty.

use prob_and_stats::kalman::Kalman1D;
use rand_distr::{Distribution, Normal};

fn main() {
    let true_position = 100.0;
    let gps_sigma = 3.0;
    let gps = Normal::new(true_position, gps_sigma).unwrap();
    let mut rng = rand::rng();

    let mut kf = Kalman1D::new(
        0.0,                   // initial x_hat: a wild guess
        1000.0,                // initial variance: very uncertain
        1.0,                   // a = 1: the car does not move
        0.0,                   // q = 0: the "stationary" model is exactly right
        gps_sigma * gps_sigma, // r = GPS variance
    );

    println!("{:>4} {:>9} {:>9} {:>9} {:>7}", "step", "gps", "x_hat", "variance", "gain");
    for step in 1..=30 {
        kf.predict();
        let y = gps.sample(&mut rng);
        let k = kf.update(y);
        if step <= 5 || step % 5 == 0 {
            println!("{:>4} {:>9.2} {:>9.2} {:>9.3} {:>7.3}", step, y, kf.x_hat, kf.variance, k);
        }
    }
}
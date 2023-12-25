use rand::{distributions::{Uniform, Distribution}, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub fn get_random_f32_vec(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let distribution = Uniform::new(-10.0f32, 10.0f32);
    let mut data = Vec::with_capacity(len);

    for _ in 0..len {
        data.push(distribution.sample(&mut rng));
    }
    
    data
}

#[allow(dead_code)]
pub fn assert_approximate(a: f32, b: f32) {
    if a.is_nan() && b.is_nan() {
        return;
    }

    if a.is_infinite() && b.is_infinite() {
        return;
    }

    if a == 0.0 || b == 0.0 {
        assert!((a - b).abs() < 0.001, "difference too big between {} and {}", a, b);
    } else {
        assert!((1.0 - (a / b)).abs() < 0.001, "difference too big between {} and {}", a, b);
    }
}

#[allow(dead_code)]
pub fn assert_approximate_vector(a: &Vec<f32>, b: &Vec<f32>) {
    for (a, b) in a.iter().zip(b.iter()) {
        assert_approximate(*a, *b);
    }
}

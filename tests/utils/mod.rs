/*
Copyright 2023 Philipp Wundrack

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use rand::{
    distributions::{Distribution, Uniform},
    Rng, SeedableRng,
};
use rand_chacha::ChaCha20Rng;

#[allow(dead_code)]
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
pub fn get_random_bool_vec(seed: u64, len: usize) -> Vec<bool> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(len);

    for _ in 0..len {
        data.push(rng.gen_bool(0.5));
    }

    data
}

#[allow(dead_code)]
pub fn assert_approximate(a: f32, b: f32, epsilon: f32) {
    if a.is_nan() && b.is_nan() {
        return;
    }

    if a.is_infinite() && b.is_infinite() {
        return;
    }

    if a == 0.0 || b == 0.0 {
        assert!(
            (a - b).abs() < epsilon,
            "difference too big between {} and {}",
            a,
            b
        );
    } else {
        assert!(
            (1.0 - (a / b)).abs() < epsilon,
            "difference too big between {} and {}",
            a,
            b
        );
    }
}

#[allow(dead_code)]
pub fn assert_approximate_vector(a: &Vec<f32>, b: &Vec<f32>, epsilon: f32) {
    for (a, b) in a.iter().zip(b.iter()) {
        assert_approximate(*a, *b, epsilon);
    }
}

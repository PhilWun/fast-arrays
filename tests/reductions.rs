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

mod utils;

use fast_arrays::Array;
use utils::{assert_approximate, get_random_f32_vec};

use rstest::rstest;

#[rstest]
#[case::sum(Array::<1>::sum, sum)]
#[case::product(Array::<1>::product, product)]
fn reduction1d_one_input(#[case] test_function: fn(&Array<1>) -> f32, #[case] target_function: fn(&Vec<f32>) -> f32) {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let array: Array<1> = data.clone().into();

        let result = test_function(&array);
        let target = target_function(&data);

        assert_approximate(result, target, 0.001);
    }
}

#[rstest]
#[case::sum(Array::<2>::sum, sum)]
#[case::product(Array::<2>::product, product)]
fn reduction2d_one_input(#[case] test_function: fn(&Array<2>) -> f32, #[case] target_function: fn(&Vec<f32>) -> f32) {
    for i in 1..32 {
        for j in 1..32 {
            let data = get_random_f32_vec(0, i * j);
            let array: Array<2> = Array::<2>::from_vec(&data, [i, j]);

            let result = test_function(&array);
            let target = target_function(&data);

            assert_approximate(result, target, 0.001);
        }
    }
}

#[rstest]
#[case::dot_product(Array::dot_product, dot_product)]
fn reduction_two_inputs(#[case] test_function: fn(&Array<1>, &Array<1>) -> f32, #[case] target_function: fn(&Vec<f32>, &Vec<f32>) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);
        let array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();

        let result = test_function(&array1, &array2);
        let target = target_function(&data1, &data2);

        assert_approximate(result, target, 0.001);
    }
}

fn sum(input: &Vec<f32>) -> f32 {
    if input.len() == 0 {
        return 0.0;
    }

    let mut sum = 0.0;

    for i in input.iter() {
        sum += i;
    }

    sum
}

fn product(input: &Vec<f32>) -> f32 {
    if input.len() == 0 {
        return 1.0;
    }

    let mut product = 1.0;

    for i in input.iter() {
        product *= i;
    }

    product
}

fn dot_product(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    if a.len() == 0 {
        return 0.0;
    }

    let mut dot_product = 0.0;

    for (d1, d2) in a.iter().zip(b.iter()) {
        dot_product += *d1 * *d2;
    }

    dot_product
}

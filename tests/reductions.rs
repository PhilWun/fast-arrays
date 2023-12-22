mod utils;

use fast_arrays::Array1D;
use utils::{assert_approximate, get_random_f32_vec};

use rstest::rstest;

#[rstest]
#[case::sum(Array1D::sum, sum)]
#[case::product(Array1D::product, product)]
fn reduction_one_input(#[case] test_function: fn(&Array1D) -> f32, #[case] target_function: fn(&Vec<f32>) -> f32) {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let array: Array1D = data.clone().into();

        let result = test_function(&array);
        let target = target_function(&data);

        assert_approximate(result, target);
    }
}

#[rstest]
#[case::dot_product(Array1D::dot_product, dot_product)]
fn reduction_two_inputs(#[case] test_function: fn(&Array1D, &Array1D) -> f32, #[case] target_function: fn(&Vec<f32>, &Vec<f32>) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);
        let array1: Array1D = data1.clone().into();
        let array2: Array1D = data2.clone().into();

        let result = test_function(&array1, &array2);
        let target = target_function(&data1, &data2);

        assert_approximate(result, target);
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

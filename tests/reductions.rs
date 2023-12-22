mod utils;

use fast_arrays::Array1D;
use utils::{assert_approximate, get_random_f32_vec};

use rstest::rstest;

#[rstest]
#[case::sum(Array1D::sum, sum)]
#[case::product(Array1D::product, product)]
fn reduction(#[case] test_function: fn(&Array1D) -> f32, #[case] target_function: fn(&Vec<f32>) -> f32) {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let array: Array1D = data.clone().into();

        let result = test_function(&array);
        let target = target_function(&data);

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

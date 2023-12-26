mod utils;

use fast_arrays::Array;
use utils::{assert_approximate, get_random_f32_vec};

use rstest::rstest;

#[rstest]
#[case::sqrt(Array::sqrt_in_place, f32::sqrt)]
#[case::square(Array::square_in_place, |x| x * x)]
#[case::abs(Array::abs_in_place, f32::abs)]
// #[case::exp(Array1D::exp_in_place, f32::exp)]
fn in_place(#[case] test_function: fn(&mut Array<1>), #[case] target_function: fn(f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let mut array1: Array<1> = data1.clone().into();

        test_function(&mut array1);
        let result: Vec<f32> = array1.into();

        for (d, r) in data1.iter().zip(result.iter()) {
            assert_approximate(*r, target_function(*d), 0.001);
        }
    }
}

#[rstest]
#[case::sqrt(Array::sqrt, f32::sqrt)]
#[case::square(Array::square, |x| x * x)]
#[case::abs(Array::abs, f32::abs)]
// #[case::exp(Array1D::exp, f32::exp)]
fn ref_out_of_place(#[case] test_function: fn(&Array<1>) -> Array<1>, #[case] target_function: fn(f32) -> f32) {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let array: Array<1> = data.clone().into();
        let result: Vec<f32> = test_function(&array).into();

        for (d, r) in data.iter().zip(result.iter()) {
            assert_approximate(*r, target_function(*d), 0.001);
        }
    }
}

mod utils;

use std::ops::{Add, Sub, Mul, Div};

use fast_arrays::Array1D;
use utils::get_random_f32_vec;

use rstest::rstest;

#[rstest]
#[case::add(Array1D::add_in_place, f32::add)]
#[case::sub(Array1D::sub_in_place, f32::sub)]
#[case::mul(Array1D::mul_in_place, f32::mul)]
#[case::div(Array1D::div_in_place, f32::div)]
#[case::max(Array1D::max_in_place, f32::max)]
#[case::min(Array1D::min_in_place, f32::min)]
fn in_place_ref(#[case] test_function: fn(&mut Array1D, &Array1D), #[case] target_function: fn(f32, f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);

        let mut array1: Array1D = data1.clone().into();
        let array2: Array1D = data2.clone().into();

        test_function(&mut array1, &array2);
        let result: Vec<f32> = array1.into();

        for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
            assert_eq!(*r, target_function(*d1, *d2));
        }
    }
}

#[rstest]
#[case::add(Array1D::add_in_place)]
#[case::sub(Array1D::sub_in_place)]
#[case::mul(Array1D::mul_in_place)]
#[case::div(Array1D::div_in_place)]
#[case::max(Array1D::max_in_place)]
#[case::min(Array1D::min_in_place)]
#[should_panic]
fn in_place_ref_shape_mismatch(#[case] test_function: fn(&mut Array1D, &Array1D)) {
    let mut array1: Array1D = get_random_f32_vec(0, 3).into();
    let array2: Array1D = get_random_f32_vec(1, 4).into();
    let _ = test_function(&mut array1, &array2);
}

#[rstest]
#[case::add(Array1D::add, f32::add)]
#[case::sub(Array1D::sub, f32::sub)]
#[case::mul(Array1D::mul, f32::mul)]
#[case::div(Array1D::div, f32::div)]
#[case::max(Array1D::max, f32::max)]
#[case::min(Array1D::min, f32::min)]
fn out_of_place_ref(#[case] test_function: fn(&Array1D, &Array1D) -> Array1D, #[case] target_function: fn(f32, f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);

        let array1: Array1D = data1.clone().into();
        let array2: Array1D = data2.clone().into();

        let result: Vec<f32> = test_function(&array1, &array2).into();

        for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
            assert_eq!(*r, target_function(*d1, *d2));
        }
    }
}

#[rstest]
#[case::add(Array1D::add)]
#[case::sub(Array1D::sub)]
#[case::mul(Array1D::mul)]
#[case::div(Array1D::div)]
#[case::max(Array1D::max)]
#[case::min(Array1D::min)]
#[should_panic]
fn out_of_place_ref_shape_mismatch(#[case] test_function: fn(&Array1D, &Array1D) -> Array1D) {
    let array1: Array1D = get_random_f32_vec(0, 3).into();
    let array2: Array1D = get_random_f32_vec(1, 4).into();
    let _ = test_function(&array1, &array2);
}

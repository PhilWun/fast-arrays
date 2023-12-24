mod utils;

use std::ops::{Add, Sub, Mul, Div};

use fast_arrays::Array;
use utils::get_random_f32_vec;

use rstest::rstest;

#[rstest]
#[case::add(Array::add_in_place, f32::add)]
#[case::sub(Array::sub_in_place, f32::sub)]
#[case::mul(Array::mul_in_place, f32::mul)]
#[case::div(Array::div_in_place, f32::div)]
#[case::max(Array::max_in_place, f32::max)]
#[case::min(Array::min_in_place, f32::min)]
fn in_place(#[case] test_function: fn(&mut Array<1>, &Array<1>), #[case] target_function: fn(f32, f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);

        let mut array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();

        test_function(&mut array1, &array2);
        let result: Vec<f32> = array1.into();

        for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
            assert_eq!(*r, target_function(*d1, *d2));
        }
    }
}

#[rstest]
#[case::add(Array::add_in_place)]
#[case::sub(Array::sub_in_place)]
#[case::mul(Array::mul_in_place)]
#[case::div(Array::div_in_place)]
#[case::max(Array::max_in_place)]
#[case::min(Array::min_in_place)]
#[should_panic]
fn in_place_shape_mismatch(#[case] test_function: fn(&mut Array<1>, &Array<1>)) {
    let mut array1: Array<1> = get_random_f32_vec(0, 3).into();
    let array2: Array<1> = get_random_f32_vec(1, 4).into();
    let _ = test_function(&mut array1, &array2);
}

#[rstest]
#[case::add(Array::add, f32::add)]
#[case::sub(Array::sub, f32::sub)]
#[case::mul(Array::mul, f32::mul)]
#[case::div(Array::div, f32::div)]
#[case::max(Array::max, f32::max)]
#[case::min(Array::min, f32::min)]
fn out_of_place(#[case] test_function: fn(&Array<1>, &Array<1>) -> Array<1>, #[case] target_function: fn(f32, f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);

        let array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();

        let result: Vec<f32> = test_function(&array1, &array2).into();

        for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
            assert_eq!(*r, target_function(*d1, *d2));
        }
    }
}

#[rstest]
#[case::add(Array::add)]
#[case::sub(Array::sub)]
#[case::mul(Array::mul)]
#[case::div(Array::div)]
#[case::max(Array::max)]
#[case::min(Array::min)]
#[should_panic]
fn out_of_place_shape_mismatch(#[case] test_function: fn(&Array<1>, &Array<1>) -> Array<1>) {
    let array1: Array<1> = get_random_f32_vec(0, 3).into();
    let array2: Array<1> = get_random_f32_vec(1, 4).into();
    let _ = test_function(&array1, &array2);
}

#[rstest]
#[case::add(Array::add_scalar_in_place, f32::add)]
#[case::sub(Array::sub_scalar_in_place, f32::sub)]
#[case::mul(Array::mul_scalar_in_place, f32::mul)]
#[case::div(Array::div_scalar_in_place, f32::div)]
fn in_place_scalar(#[case] test_function: fn(&mut Array<1>, f32), #[case] target_function: fn(f32, f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let scalar = 4.2f32;

        let mut array1: Array<1> = data1.clone().into();

        test_function(&mut array1, scalar);
        let result: Vec<f32> = array1.into();

        for (d, r) in data1.iter().zip(result.iter()) {
            assert_eq!(*r, target_function(*d, scalar));
        }
    }
}

#[rstest]
#[case::add(Array::add_scalar, f32::add)]
#[case::sub(Array::sub_scalar, f32::sub)]
#[case::mul(Array::mul_scalar, f32::mul)]
#[case::div(Array::div_scalar, f32::div)]
fn out_of_place_scalar(#[case] test_function: fn(&Array<1>, f32) -> Array<1>, #[case] target_function: fn(f32, f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let scalar = 4.2f32;

        let array1: Array<1> = data1.clone().into();

        let result: Vec<f32> = test_function(&array1, scalar).into();

        for (d, r) in data1.iter().zip(result.iter()) {
            assert_eq!(*r, target_function(*d, scalar));
        }
    }
}

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

use fast_arrays::{Array, Mask};
use utils::get_random_f32_vec;

use rstest::rstest;

#[rstest]
#[case::eq(Array::compare_equal, f32::eq)]
#[case::neq(Array::compare_not_equal, f32::ne)]
#[case::gt(Array::compare_greater_than, f32::gt)]
#[case::ge(Array::compare_greater_than_or_equal, f32::ge)]
#[case::lt(Array::compare_less_than, f32::lt)]
#[case::le(Array::compare_less_than_or_equal, f32::le)]
fn comparison(#[case] test_function: fn(&Array<1>, &Array<1>) -> Mask<1>, #[case] target_function: fn(&f32, &f32) -> bool) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);

        let array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();

        let result: Vec<bool> = test_function(&array1, &array2).into();

        for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
            assert_eq!(*r, target_function(d1, d2));
        }
    }
}

#[rstest]
#[case::eq(Array::compare_equal, f32::eq)]
#[case::neq(Array::compare_not_equal, f32::ne)]
#[case::gt(Array::compare_greater_than, f32::gt)]
#[case::ge(Array::compare_greater_than_or_equal, f32::ge)]
#[case::lt(Array::compare_less_than, f32::lt)]
#[case::le(Array::compare_less_than_or_equal, f32::le)]
#[should_panic]
fn comparison_mismatched_shapes(#[case] test_function: fn(&Array<1>, &Array<1>) -> Mask<1>, #[case] target_function: fn(&f32, &f32) -> bool) {
    let data1 = get_random_f32_vec(0, 3);
    let data2 = get_random_f32_vec(1, 4);

    let array1: Array<1> = data1.clone().into();
    let array2: Array<1> = data2.clone().into();

    let result: Vec<bool> = test_function(&array1, &array2).into();

    for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
        assert_eq!(*r, target_function(d1, d2));
    }
}

#[rstest]
#[case::eq(Array::compare_equal_in_place, f32::eq)]
#[case::neq(Array::compare_not_equal_in_place, f32::ne)]
#[case::gt(Array::compare_greater_than_in_place, f32::gt)]
#[case::ge(Array::compare_greater_than_or_equal_in_place, f32::ge)]
#[case::lt(Array::compare_less_than_in_place, f32::lt)]
#[case::le(Array::compare_less_than_or_equal_in_place, f32::le)]
fn comparison_in_place(#[case] test_function: fn(&Array<1>, &Array<1>, &mut Mask<1>), #[case] target_function: fn(&f32, &f32) -> bool) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);

        let array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();

        let mut mask = Mask::<1>::new(i);
        test_function(&array1, &array2, &mut mask);

        let result: Vec<bool> = mask.into();

        for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
            assert_eq!(*r, target_function(d1, d2));
        }
    }
}

#[rstest]
#[case::eq(Array::compare_equal_in_place, f32::eq)]
#[case::neq(Array::compare_not_equal_in_place, f32::ne)]
#[case::gt(Array::compare_greater_than_in_place, f32::gt)]
#[case::ge(Array::compare_greater_than_or_equal_in_place, f32::ge)]
#[case::lt(Array::compare_less_than_in_place, f32::lt)]
#[case::le(Array::compare_less_than_or_equal_in_place, f32::le)]
#[should_panic]
fn comparison_mismatched_shapes_in_place(#[case] test_function: fn(&Array<1>, &Array<1>, &mut Mask<1>), #[case] target_function: fn(&f32, &f32) -> bool) {
    let data1 = get_random_f32_vec(0, 3);
    let data2 = get_random_f32_vec(1, 4);

    let array1: Array<1> = data1.clone().into();
    let array2: Array<1> = data2.clone().into();

    let mut mask = Mask::<1>::new(5);
    test_function(&array1, &array2, &mut mask);

    let result: Vec<bool> = mask.into();

    for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
        assert_eq!(*r, target_function(d1, d2));
    }
}

#[rstest]
#[case::eq(Array::compare_scalar_equal, f32::eq)]
#[case::neq(Array::compare_scalar_not_equal, f32::ne)]
#[case::gt(Array::compare_scalar_greater_than, f32::gt)]
#[case::ge(Array::compare_scalar_greater_than_or_equal, f32::ge)]
#[case::lt(Array::compare_scalar_less_than, f32::lt)]
#[case::le(Array::compare_scalar_less_than_or_equal, f32::le)]
fn comparison_scalar(#[case] test_function: fn(&Array<1>, f32) -> Mask<1>, #[case] target_function: fn(&f32, &f32) -> bool) {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);

        let array: Array<1> = data.clone().into();

        let result: Vec<bool> = test_function(&array, 0.0).into();

        for (d, r) in data.iter().zip(result.iter()) {
            assert_eq!(*r, target_function(d, &0.0));
        }
    }
}

#[rstest]
#[case::eq(Array::compare_scalar_equal_in_place, f32::eq)]
#[case::neq(Array::compare_scalar_not_equal_in_place, f32::ne)]
#[case::gt(Array::compare_scalar_greater_than_in_place, f32::gt)]
#[case::ge(Array::compare_scalar_greater_than_or_equal_in_place, f32::ge)]
#[case::lt(Array::compare_scalar_less_than_in_place, f32::lt)]
#[case::le(Array::compare_scalar_less_than_or_equal_in_place, f32::le)]
fn comparison_scalar_in_place(#[case] test_function: fn(&Array<1>, f32, &mut Mask<1>), #[case] target_function: fn(&f32, &f32) -> bool) {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);

        let array: Array<1> = data.clone().into();
        let mut mask = Mask::<1>::new(i);
        test_function(&array, 0.0, &mut mask);

        let result: Vec<bool> = mask.into();

        for (d, r) in data.iter().zip(result.iter()) {
            assert_eq!(*r, target_function(d, &0.0));
        }
    }
}

#[rstest]
#[case::eq(Array::compare_scalar_equal_in_place, f32::eq)]
#[case::neq(Array::compare_scalar_not_equal_in_place, f32::ne)]
#[case::gt(Array::compare_scalar_greater_than_in_place, f32::gt)]
#[case::ge(Array::compare_scalar_greater_than_or_equal_in_place, f32::ge)]
#[case::lt(Array::compare_scalar_less_than_in_place, f32::lt)]
#[case::le(Array::compare_scalar_less_than_or_equal_in_place, f32::le)]
#[should_panic]
fn comparison_scalar_in_place_mismatched_shape(#[case] test_function: fn(&Array<1>, f32, &mut Mask<1>), #[case] target_function: fn(&f32, &f32) -> bool) {
    let data = get_random_f32_vec(0, 3);

    let array: Array<1> = data.clone().into();
    let mut mask = Mask::<1>::new(4);
    test_function(&array, 0.0, &mut mask);

    let result: Vec<bool> = mask.into();

    for (d, r) in data.iter().zip(result.iter()) {
        assert_eq!(*r, target_function(d, &0.0));
    }
}

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

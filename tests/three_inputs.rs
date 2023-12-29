/**
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
#[case::fmadd(Array::fmadd_in_place, |x, y, z| y * z + x)]
fn in_place_ref(#[case] test_function: fn(&mut Array<1>, &Array<1>, &Array<1>), #[case] target_function: fn(f32, f32, f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);
        let data3 = get_random_f32_vec(2, i);

        let mut array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();
        let array3: Array<1> = data3.clone().into();

        test_function(&mut array1, &array2, &array3);
        let result: Vec<f32> = array1.into();

        for (((d1, d2), d3), r) in data1.iter().zip(data2.iter()).zip(data3.iter()).zip(result.iter()) {
            assert_approximate(*r, target_function(*d1, *d2, *d3), 0.001);
        }
    }
}

#[rstest]
#[case::fmadd(Array::fmadd_in_place)]
#[should_panic]
fn in_place_ref_shape_mismatch(#[case] test_function: fn(&mut Array<1>, &Array<1>, &Array<1>)) {
    let mut a: Array<1> = get_random_f32_vec(0, 3).into();
    let b: Array<1> = get_random_f32_vec(1, 4).into();
    let c: Array<1> = get_random_f32_vec(2, 5).into();       
    
    test_function(&mut a, &b, &c);
}

#[rstest]
#[case::fmadd(Array::fmadd, |x, y, z| y * z + x)]
fn out_of_place_ref(#[case] test_function: fn(&Array<1>, &Array<1>, &Array<1>) -> Array<1>, #[case] target_function: fn(f32, f32, f32) -> f32) {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);
        let data3 = get_random_f32_vec(2, i);

        let array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();
        let array3: Array<1> = data3.clone().into();

        let result: Vec<f32> = test_function(&array1, &array2, &array3).into();

        for (((d1, d2), d3), r) in data1.iter().zip(data2.iter()).zip(data3.iter()).zip(result.iter()) {
            assert_approximate(*r, target_function(*d1, *d2, *d3), 0.001);
        }
    }
}

#[rstest]
#[case::fmadd(Array::fmadd)]
#[should_panic]
fn out_of_place_ref_shape_mismatch(#[case] test_function: fn(&Array<1>, &Array<1>, &Array<1>) -> Array<1>) {
    let a: Array<1> = get_random_f32_vec(0, 3).into();
    let b: Array<1> = get_random_f32_vec(1, 4).into();
    let c: Array<1> = get_random_f32_vec(2, 5).into();

    test_function(&a, &b, &c);
}

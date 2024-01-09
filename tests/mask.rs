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

use fast_arrays::Mask;
use rstest::rstest;

use crate::utils::get_random_bool_vec;

#[test]
fn convert() {
    for i in 0..64 {
        let data = get_random_bool_vec(0, i);
        let converted: Mask<1> = data.clone().into();
        let converted_back: Vec<bool> = converted.into();

        assert_eq!(converted_back, data);
    }
}

#[rstest]
#[case::not(Mask::not_in_place, |x: bool| !x)]
fn in_place(#[case] test_function: fn(&mut Mask<1>), #[case] target_function: fn(bool) -> bool) {
    for i in 0..64 {
        let data1 = get_random_bool_vec(0, i);
        let mut array1: Mask<1> = data1.clone().into();

        test_function(&mut array1);
        let result: Vec<bool> = array1.into();

        for (d, r) in data1.iter().zip(result.iter()) {
            assert_eq!(*r, target_function(*d));
        }
    }
}

#[rstest]
#[case::and(Mask::and_in_place, |a, b| a & b)]
#[case::or(Mask::or_in_place, |a, b| a | b)]
fn two_inputs(#[case] test_function: fn(&mut Mask<1>, &Mask<1>), #[case] target_function: fn(bool, bool) -> bool) {
    for i in 0..64 {
        let data1 = get_random_bool_vec(0, i);
        let data2 = get_random_bool_vec(1, i);

        let mut array1: Mask<1> = data1.clone().into();
        let array2: Mask<1> = data2.clone().into();

        test_function(&mut array1, &array2);
        let result: Vec<bool> = array1.into();

        for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
            assert_eq!(*r, target_function(*d1, *d2));
        }
    }
}

#[test]
fn get() {
    for i in 1..64 {
        let data = get_random_bool_vec(0, i);
        let converted: Mask<1> = data.clone().into();
        
        for j in 0..i {
            assert_eq!(data[j], converted.get(j));
        }
    }
}

#[test]
fn tile_in_place() {
    for i in 1..5 {
        for j in 1..5 {
            let n = i * 16;
            let k = j * 16;
            let data = get_random_bool_vec(0, n);
            let mask: Mask<1> = data.clone().into();
            let mut output = Mask::new(n * k);

            mask.tile_in_place(k, &mut output);

            let output_data: Vec<bool> = output.into();

            for (i, m) in output_data.iter().enumerate() {
                assert_eq!(*m, data[i % n]);
            }
        }
    }
}

#[test]
fn repeat_in_place() {
    for i in 1..5 {
        for j in 1..5 {
            let n = i * 16;
            let k = j * 16;
            let data = get_random_bool_vec(0, n);
            let mask: Mask<1> = data.clone().into();
            let mut output = Mask::new(n * k);

            mask.repeat_in_place(k, &mut output);

            let output_data: Vec<bool> = output.into();

            for (i, m) in output_data.iter().enumerate() {
                assert_eq!(*m, data[i / k]);
            }
        }
    }
}

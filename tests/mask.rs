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
        converted.assert_invariants_satisfied();
        let converted_back: Vec<bool> = converted.into();

        assert_eq!(converted_back, data);
    }
}

#[test]
fn copy() {
    for i in 0..64 {
        let data = get_random_bool_vec(0, i);
        let mask1: Mask<1> = data.clone().into();
        mask1.assert_invariants_satisfied();
        
        let mut mask2 = Mask::<1>::zeros(mask1.get_shape());
        mask2.copy(&mask1);

        let copied_data: Vec<bool> = mask2.into();

        assert_eq!(copied_data, data);
    }
}

#[test]
fn serde1d() {
    for i in 0..64 {
        let data = get_random_bool_vec(0, i);
        let converted: Mask<1> = data.clone().into();

        let json = serde_json::to_string(&converted).unwrap();
        let deserialized: Mask<1> = serde_json::from_str(&json).unwrap();

        assert_eq!(converted.get_shape(), deserialized.get_shape());

        let converted_back: Vec<bool> = deserialized.into();

        assert_eq!(converted_back, data);
    }
}

#[test]
fn serde2d() {
    for i in 1..32 {
        for j in 1..32 {
            let data = get_random_bool_vec(0, i * j);
            let converted: Mask<2> = Mask::<2>::from_vec(&data, [i, j]);

            let json = serde_json::to_string(&converted).unwrap();
            let deserialized: Mask<2> = serde_json::from_str(&json).unwrap();

            assert_eq!(converted.get_shape(), deserialized.get_shape());

            for i2 in 0..i {
                for j2 in 0..j {
                    assert_eq!(converted.get(i2, j2), deserialized.get(i2, j2));
                }
            }
        }
    }
}

#[rstest]
#[case::not(Mask::not_in_place, |x: bool| !x)]
fn in_place(#[case] test_function: fn(&mut Mask<1>), #[case] target_function: fn(bool) -> bool) {
    for i in 0..64 {
        let data1 = get_random_bool_vec(0, i);
        let mut mask: Mask<1> = data1.clone().into();
        mask.assert_invariants_satisfied();

        test_function(&mut mask);
        mask.assert_invariants_satisfied();

        let result: Vec<bool> = mask.into();

        for (d, r) in data1.iter().zip(result.iter()) {
            assert_eq!(*r, target_function(*d));
        }
    }
}

#[rstest]
#[case::and(Mask::and_in_place, |a, b| a & b)]
#[case::or(Mask::or_in_place, |a, b| a | b)]
fn two_inputs(
    #[case] test_function: fn(&mut Mask<1>, &Mask<1>),
    #[case] target_function: fn(bool, bool) -> bool,
) {
    for i in 0..64 {
        let data1 = get_random_bool_vec(0, i);
        let data2 = get_random_bool_vec(1, i);

        let mut mask1: Mask<1> = data1.clone().into();
        mask1.assert_invariants_satisfied();
        let mask2: Mask<1> = data2.clone().into();
        mask2.assert_invariants_satisfied();

        test_function(&mut mask1, &mask2);
        mask1.assert_invariants_satisfied();
        let result: Vec<bool> = mask1.into();

        for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
            assert_eq!(*r, target_function(*d1, *d2));
        }
    }
}

#[rstest]
#[case::and(Mask::and_in_place)]
#[case::or(Mask::or_in_place)]
#[should_panic]
fn two_inputs_mismatched_shape(#[case] test_function: fn(&mut Mask<1>, &Mask<1>)) {
    let data1 = get_random_bool_vec(0, 4);
    let data2 = get_random_bool_vec(1, 5);

    let mut mask1: Mask<1> = data1.clone().into();
    mask1.assert_invariants_satisfied();
    let mask2: Mask<1> = data2.clone().into();
    mask2.assert_invariants_satisfied();

    test_function(&mut mask1, &mask2);
    mask1.assert_invariants_satisfied();
}

#[test]
fn get() {
    for i in 1..64 {
        let data = get_random_bool_vec(0, i);
        let converted: Mask<1> = data.clone().into();
        converted.assert_invariants_satisfied();

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
            mask.assert_invariants_satisfied();
            let mut output = Mask::zeros(&[n * k]);
            output.assert_invariants_satisfied();

            mask.tile_in_place(k, &mut output);
            output.assert_invariants_satisfied();

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
            mask.assert_invariants_satisfied();
            let mut output = Mask::zeros(&[n * k]);
            output.assert_invariants_satisfied();

            mask.repeat_in_place(k, &mut output);
            output.assert_invariants_satisfied();

            let output_data: Vec<bool> = output.into();

            for (i, m) in output_data.iter().enumerate() {
                assert_eq!(*m, data[i / k]);
            }
        }
    }
}

#[test]
fn repeat_as_row_in_place() {
    for columns in 1..32 {
        for rows in 1..32 {
            let data = get_random_bool_vec(0, columns);
            let mask_1d: Mask<1> = data.clone().into();
            let mut output = Mask::zeros(&[rows, columns]);

            mask_1d.repeat_as_row_in_place(rows, &mut output);

            for i in 0..columns {
                for j in 0..rows {
                    assert_eq!(output.get(j, i), data[i]);
                }
            }
        }
    }
}

#[test]
fn repeat_as_row_in_place_shape_mismatch() {
    for columns in 1..32 {
        for rows in 1..32 {
            let data = get_random_bool_vec(0, rows);
            let array_1d: Mask<1> = data.clone().into();
            let mut output = Mask::zeros(&[rows, columns]);

            array_1d.repeat_as_column_in_place(columns, &mut output);

            for k in 0..columns {
                for l in 0..rows {
                    assert_eq!(output.get(l, k), data[l]);
                }
            }
        }
    }
}

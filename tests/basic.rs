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
use utils::{get_random_f32_vec, get_random_bool_vec};

#[test]
fn conversion1d() {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let converted: Array<1> = data.clone().into();
        let converted_back: Vec<f32> = converted.into();

        assert_eq!(converted_back, data);
    }
}

#[test]
fn conversion2d() {
    for i in 1..32 {
        for j in 1..32 {
            let data = get_random_f32_vec(0, i * j);
            let converted: Array<2> = Array::<2>::from_vec(&data, [i, j]);
            let converted_back: Vec<f32> = converted.into();

            assert_eq!(converted_back, data);
        }
    }
}

#[test]
fn zeros1d() {
    for i in 0..64 {
        let zeros: Vec<f32> = Array::zeros(&[i]).into();

        assert_eq!(zeros, vec![0.0; i]);
    }
}

#[test]
fn zeros2d() {
    for i in 0..32 {
        for j in 0..32 {
            let zeros: Vec<f32> = Array::zeros(&[i, j]).into();

            assert_eq!(zeros, vec![0.0; i * j]);
        }
    }
}

#[test]
fn from_value1d() {
    for i in 0..64 {
        let array: Vec<f32> = Array::new_from_value(&[i], 42.0).into();

        assert_eq!(array, vec![42.0; i]);
    }
}

#[test]
fn from_value2d() {
    for i in 0..32 {
        for j in 0..32 {
            let array: Vec<f32> = Array::new_from_value(&[i, j], 42.0).into();

            assert_eq!(array, vec![42.0; i * j]);
        }
    }
}

#[test]
fn random1d() {
    for i in 0..64 {
        let array: Vec<f32> = Array::random_uniform(&[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).into();

        for v in array {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }
}

#[test]
fn random2d() {
    for i in 0..32 {
        for j in 0..32 {
            let array: Vec<f32> = Array::random_uniform(&[i, j], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).into();

            for v in array {
                assert!(v >= 0.0 && v <= 1.0);
            }
        }
    }
}

#[test]
fn get() {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let array: Array<1> = data.clone().into();

        for j in 0..i {
            assert_eq!(array.get(j), *data.get(j).unwrap());
        }
    }
}

#[test]
#[should_panic]
fn get_out_of_bounds() {
    let data = get_random_f32_vec(0, 64);
    let array: Array<1> = data.into();

    array.get(64);
}

#[test]
fn set_1d() {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let mut array: Array<1> = data.clone().into();

        for j in 0..i {
            array.set(j, (j + 10) as f32);
            assert_eq!(array.get(j), (j + 10) as f32);
        }
    }
}

#[test]
#[should_panic]
fn set_1d_out_of_bounds() {
    let data = get_random_f32_vec(0, 64);
    let mut array: Array<1> = data.into();
    array.set(64, 42.0);
}

#[test]
fn get_set_2d() {
    for rows in 1..32 {
        for columns in 1..32 {
            let data = get_random_f32_vec(0, rows * columns);
            let mut array: Array<2> = Array::zeros(&[rows, columns]);

            for row in 0..rows {
                for column in 0..columns {
                    array.set(row, column, data[row * columns + column]);
                }
            }

            for row in 0..rows {
                for column in 0..columns {
                    assert_eq!(array.get(row, column), data[row * columns + column]);
                }
            }
        }
    }
}

#[test]
#[should_panic]
fn get_2d_out_of_bounds() {
    let array = Array::zeros(&[3, 4]);
    array.get(4, 4);
}

#[test]
#[should_panic]
fn set_2d_out_of_bounds() {
    let mut array = Array::zeros(&[3, 4]);
    array.set(4, 4, 42.0);
}

#[test]
fn set_all() {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let mut array: Array<1> = data.clone().into();

        array.set_all(42.0);
        let new_data: Vec<f32> = array.into();

        assert_eq!(new_data, vec![42.0f32; i]);
    }
}

#[test]
fn set_masked() {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let mask_data = get_random_bool_vec(1, i);
        let mut array: Array<1> = data.clone().into();
        let mask: Mask<1> = mask_data.clone().into();

        array.set_masked(42.0, &mask);

        let result: Vec<f32> = array.into();

        for ((r, d), m) in result.iter().zip(data.iter()).zip(mask_data.iter()) {
            if *m {
                assert_eq!(*r, 42.0);
            } else {
                assert_eq!(r, d);
            }
        }
    }
}

#[test]
#[should_panic]
fn set_masked_mismatched_shapes() {
    let data = get_random_f32_vec(0, 3);
    let mask_data = get_random_bool_vec(1, 4);
    let mut array: Array<1> = data.clone().into();
    let mask: Mask<1> = mask_data.clone().into();

    array.set_masked(42.0, &mask);
}

#[test]
fn set_masked2() {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let mask_data = get_random_bool_vec(1, i);

        let mut array1: Array<1> = data1.clone().into();
        let mask: Mask<1> = mask_data.clone().into();

        array1.set_masked2(42.0, 43.0, &mask);

        let result: Vec<f32> = array1.into();

        for (r, m) in result.iter().zip(mask_data.iter()) {
            if *m {
                assert_eq!(*r, 43.0);
            } else {
                assert_eq!(*r, 42.0);
            }
        }
    }
}

#[test]
#[should_panic]
fn set_masked2_mismatched_shapes() {
    let data1 = get_random_f32_vec(0, 3);
    let mask_data = get_random_bool_vec(1, 4);

    let mut array1: Array<1> = data1.clone().into();
    let mask: Mask<1> = mask_data.clone().into();

    array1.set_masked2(42.0, 43.0, &mask);
}

#[test]
fn copy() {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);

        let mut array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();

        array1.copy(&array2);

        let result: Vec<f32> = array1.into();

        for (r, d) in result.iter().zip(data2.iter()) {
            assert_eq!(r, d);
        }
    }
}

#[test]
#[should_panic]
fn copy_mismatched_shapes() {
    let data1 = get_random_f32_vec(0, 3);
    let data2 = get_random_f32_vec(1, 4);

    let mut array1: Array<1> = data1.clone().into();
    let array2: Array<1> = data2.clone().into();

    array1.copy(&array2);
}

#[test]
fn copy_masked() {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);
        let mask_data = get_random_bool_vec(2, i);

        let mut array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();
        let mask: Mask<1> = mask_data.clone().into();

        array1.copy_masked(&array2, &mask);

        let result: Vec<f32> = array1.into();

        for (((r, m), d1), d2) in result.iter().zip(mask_data.iter()).zip(data1.iter()).zip(data2.iter()) {
            if *m {
                assert_eq!(*r, *d2);
            } else {
                assert_eq!(*r, *d1);
            }
        }
    }
}

#[test]
#[should_panic]
fn copy_masked_mismatched_shapes() {
    let data1 = get_random_f32_vec(0, 3);
    let data2 = get_random_f32_vec(1, 4);
    let mask_data = get_random_bool_vec(2, 5);

    let mut array1: Array<1> = data1.clone().into();
    let array2: Array<1> = data2.clone().into();
    let mask: Mask<1> = mask_data.clone().into();

    array1.copy_masked(&array2, &mask);
}

#[test]
fn copy_masked2() {
    for i in 0..64 {
        let data1 = get_random_f32_vec(0, i);
        let data2 = get_random_f32_vec(1, i);
        let data3 = get_random_f32_vec(2, i);
        let mask_data = get_random_bool_vec(3, i);

        let mut array1: Array<1> = data1.clone().into();
        let array2: Array<1> = data2.clone().into();
        let array3: Array<1> = data3.clone().into();
        let mask: Mask<1> = mask_data.clone().into();

        array1.copy_masked2(&array2, &array3, &mask);

        let result: Vec<f32> = array1.into();

        for (((r, m), d2), d3) in result.iter().zip(mask_data.iter()).zip(data2.iter()).zip(data3.iter()) {
            if *m {
                assert_eq!(*r, *d3);
            } else {
                assert_eq!(*r, *d2);
            }
        }
    }
}

#[test]
#[should_panic]
fn copy_masked2_mismatched_shapes() {
    let data1 = get_random_f32_vec(0, 2);
    let data2 = get_random_f32_vec(1, 3);
    let data3 = get_random_f32_vec(2, 4);
    let mask_data = get_random_bool_vec(3, 5);

    let mut array1: Array<1> = data1.clone().into();
    let array2: Array<1> = data2.clone().into();
    let array3: Array<1> = data3.clone().into();
    let mask: Mask<1> = mask_data.clone().into();

    array1.copy_masked2(&array2, &array3, &mask);
}

#[test]
fn tile_in_place() {
    for i in 1..5 {
        for j in 1..5 {
            let n = i * 16;
            let k = j * 16;
            let data = get_random_f32_vec(0, n);
            let array: Array<1> = data.clone().into();
            let mut output = Array::zeros(&[n * k]);

            array.tile_in_place(k, &mut output);

            let output_data: Vec<f32> = output.into();

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
            let data = get_random_f32_vec(0, n);
            let array: Array<1> = data.clone().into();
            let mut output = Array::zeros(&[n * k]);

            array.repeat_in_place(k, &mut output);

            let output_data: Vec<f32> = output.into();

            for (i, m) in output_data.iter().enumerate() {
                assert_eq!(*m, data[i / k]);
            }
        }
    }
}

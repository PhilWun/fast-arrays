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

use fast_arrays::Array;
use utils::get_random_f32_vec;

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
        let array: Vec<f32> = Array::random_uniform(&[i], -1.0, 1.0, Some(0)).into();

        for v in array {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }
}

#[test]
fn random2d() {
    for i in 0..32 {
        for j in 0..32 {
            let array: Vec<f32> = Array::random_uniform(&[i, j], -1.0, 1.0, Some(0)).into();

            for v in array {
                assert!(v >= -1.0 && v <= 1.0);
            }
        }
    }
}

#[test]
fn get() {
    let data = get_random_f32_vec(0, 64);
    let array: Array<1> = data.clone().into();

    for i in 0..64 {
        assert_eq!(array.get(i), *data.get(i).unwrap());
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
fn set() {
    let data = get_random_f32_vec(0, 64);
    let mut array: Array<1> = data.clone().into();

    for i in 0..64 {
        array.set(i, (i + 10) as f32);
        assert_eq!(array.get(i), (i + 10) as f32);
    }
}

#[test]
#[should_panic]
fn set_out_of_bounds() {
    let data = get_random_f32_vec(0, 64);
    let mut array: Array<1> = data.into();
    array.set(64, 42.0);
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

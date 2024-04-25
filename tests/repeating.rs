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
fn repeat_as_row_in_place() {
    for columns in 1..32 {
        for rows in 1..32 {
            let data = get_random_f32_vec(0, columns);
            let array_1d: Array<1, _> = data.clone().into();
            let mut output = Array::zeros(&[rows, columns]);

            array_1d.repeat_as_row_in_place(rows, &mut output);

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
            let data = get_random_f32_vec(0, rows);
            let array_1d: Array<1, _> = data.clone().into();
            let mut output = Array::zeros(&[rows, columns]);

            array_1d.repeat_as_column_in_place(columns, &mut output);

            for k in 0..columns {
                for l in 0..rows {
                    assert_eq!(output.get(l, k), data[l]);
                }
            }
        }
    }
}

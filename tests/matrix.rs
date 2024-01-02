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
use utils::{assert_approximate_vector, get_random_f32_vec};

#[test]
fn matrix_vector_multiplication() {
    for i in 1..32 {
        for j in 1..32 {
            let matrix_data = get_random_f32_vec(0, i * j);
            let matrix: Array<2> = Array::<2>::from_vec(&matrix_data, [i, j]);
            let vector_data = get_random_f32_vec(1, j);
            let vector: Array<1> = vector_data.clone().into();

            let result: Vec<f32> = matrix.vector_multiplication(&vector).into();
            let target = matrix_vector_multiplication_reference(&matrix_data, i, j, &vector_data);

            assert_approximate_vector(&result, &target, 0.001);
        }
    }
}

fn matrix_vector_multiplication_reference(matrix: &Vec<f32>, rows: usize, columns: usize, vector: &Vec<f32>) -> Vec<f32> {
    let mut result = Vec::with_capacity(columns);

    for i in 0..rows {
        let mut sum = 0.0;

        for j in 0..columns {
            sum += matrix[i * columns + j] * vector[j];
        }

        result.push(sum);
    }

    result
}

#[test]
fn matrix_matrix_multiplication() {
    for i in 1..32 {
        for j in 1..32 {
            for k in 1..32 {
                let matrix_a_data = get_random_f32_vec(0, i * j);
                let matrix_a: Array<2> = Array::<2>::from_vec(&matrix_a_data, [i, j]);
                let matrix_b_data = get_random_f32_vec(0, j * k);
                let matrix_b: Array<2> = Array::<2>::from_vec(&matrix_b_data, [j, k]);

                let result: Vec<f32> = matrix_a.matrix_multiplication(&matrix_b).into();
                let target = matrix_matrix_multiplication_reference(&matrix_a_data, i, j, &matrix_b_data, k);

                assert_approximate_vector(&result, &target, 0.1);
            }
        }
    }
}

fn matrix_matrix_multiplication_reference(a: &Vec<f32>, a_rows: usize, a_columns: usize, b: &Vec<f32>, b_columns: usize) -> Vec<f32> {
    let mut result = vec![0.0; a_rows * b_columns];

    for a_row in 0..a_rows {
        for b_column in 0..b_columns {
            for inner_loop_index in 0..a_columns {
                result[a_row * b_columns + b_column] += a[a_row * a_columns + inner_loop_index] * b[inner_loop_index * b_columns + b_column];
            }
        }
    }

    result
}

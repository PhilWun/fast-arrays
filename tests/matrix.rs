mod utils;

use fast_arrays::Array;
use utils::{assert_approximate_vector, get_random_f32_vec};

#[test]
fn matrix_vector_multiplication() {
    for i in 1..32 {
        for j in 17..32 {
            let matrix_data = get_random_f32_vec(0, i * j);
            let matrix: Array<2> = Array::<2>::from_vec(&matrix_data, [i, j]);
            let vector_data = get_random_f32_vec(1, j);
            let vector: Array<1> = vector_data.clone().into();

            let result: Vec<f32> = matrix.vector_multiplication(&vector).into();
            let target = matrix_vector_multiplication_reference(&matrix_data, i, j, &vector_data);

            assert_approximate_vector(&result, &target);
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

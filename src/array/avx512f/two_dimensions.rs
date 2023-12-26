use std::arch::x86_64::{_mm512_add_ps, _mm512_mask_add_ps, _mm512_mul_ps, _mm512_mask_mul_ps, _mm512_reduce_add_ps, _mm512_reduce_mul_ps, _mm512_fmadd_ps, _mm512_mask3_fmadd_ps, __m512};

use crate::Array;

use super::{m512_to_array, array_to_m512, reduce};

impl From<Array<2>> for Vec<f32> {
    fn from(value: Array<2>) -> Self {
        let mut converted = Vec::with_capacity(value.shape[0] * value.shape[1]);
        let column_count = value.shape[1];
        let registers_per_row = column_count.div_ceil(16);
        
        for (i, register) in value.data.iter().enumerate() {
            let register = m512_to_array(*register);
            let mut limit = 16;
            
            // if it is the last register in the row
            if (i + 1) % registers_per_row == 0 {
                limit = (column_count - 1) % 16 + 1;
            }

            for j in 0..limit {
                converted.push(register[j]);
            }
        }

        converted
    }
}

impl Array<2> {
    pub fn from_vec(data: &Vec<f32>, shape: [usize; 2]) -> Self {
        assert!(shape[0] > 0);
        assert!(shape[1] > 0);

        let row_count = shape[0];
        let column_count = shape[1];
        let registers_per_row = column_count.div_ceil(16);
        let mut new_data = Vec::with_capacity(registers_per_row * row_count);
        let mut index = 0;

        for _ in 0..row_count {
            for c in 0..registers_per_row {
                let mut register = [0.0f32; 16];

                let mut limit = 16;
            
                // if it is the last register in the row
                if (c + 1) % registers_per_row == 0 {
                    limit = (column_count - 1) % 16 + 1;
                }

                for i in 0..limit {
                    register[i] = data[index];
                    index += 1;
                }

                new_data.push(array_to_m512(register));
            }
        }

        Self {
            data: new_data,
            shape
        }
    }

    pub fn sum(&self) -> f32 {
        assert!(self.shape[0] > 0);
        assert!(self.shape[1] > 0);

        let row_count = self.shape[0];
        let column_count = self.shape[1];
        let registers_per_row = column_count.div_ceil(16);

        unsafe {
            let mut sum_register = array_to_m512([0.0; 16]);

            for i in 0..row_count {
                let start = i * registers_per_row;
                let end = start + registers_per_row;

                let intermediate_result = reduce(&self.data[start..end], column_count, 0.0, _mm512_add_ps, _mm512_mask_add_ps);

                sum_register = _mm512_add_ps(sum_register, intermediate_result);
            }
            
            _mm512_reduce_add_ps(sum_register)
        }
    }

    pub fn product(&self) -> f32 {
        assert!(self.shape[0] > 0);
        assert!(self.shape[1] > 0);

        let row_count = self.shape[0];
        let column_count = self.shape[1];
        let registers_per_row = column_count.div_ceil(16);

        unsafe {
            let mut product_register = array_to_m512([1.0; 16]);

            for i in 0..row_count {
                let start = i * registers_per_row;
                let end = start + registers_per_row;

                let intermediate_result = reduce(&self.data[start..end], column_count, 1.0, _mm512_mul_ps, _mm512_mask_mul_ps);

                product_register = _mm512_mul_ps(product_register, intermediate_result);
            }
            
            _mm512_reduce_mul_ps(product_register)
        }
    }

    pub fn vector_multiplication(&self, other: &Array<1>) -> Array<1> {
        assert_eq!(self.shape[1], other.shape[0]);

        let row_count = self.shape[0];
        let column_count = self.shape[1];
        let registers_per_row = column_count.div_ceil(16);
        let mut last_register_mask = 0xFFFF;

        if column_count % 16 != 0 {
            last_register_mask = 0xFFFF >> (16 - (column_count % 16));
        }

        let mut result = Vec::with_capacity(row_count);

        unsafe {
            for i in 0..row_count {
                let mut sum = array_to_m512([0.0; 16]);

                for j in 0..registers_per_row - 1 {
                    sum = _mm512_fmadd_ps(self.data[i * registers_per_row + j], other.data[j], sum);
                }

                sum = _mm512_mask3_fmadd_ps(self.data[(i + 1) * registers_per_row - 1], other.data[registers_per_row - 1], sum, last_register_mask);

                result.push(_mm512_reduce_add_ps(sum));
            }
        }

        result.into()
    }

    fn transpose_chunk(chunk: &[__m512; 16]) -> [__m512; 16] {
        let split_chunk = [
            m512_to_array(chunk[0]),
            m512_to_array(chunk[1]),
            m512_to_array(chunk[2]),
            m512_to_array(chunk[3]),
            m512_to_array(chunk[4]),
            m512_to_array(chunk[5]),
            m512_to_array(chunk[6]),
            m512_to_array(chunk[7]),
            m512_to_array(chunk[8]),
            m512_to_array(chunk[9]),
            m512_to_array(chunk[10]),
            m512_to_array(chunk[11]),
            m512_to_array(chunk[12]),
            m512_to_array(chunk[13]),
            m512_to_array(chunk[14]),
            m512_to_array(chunk[15]),
        ];
        let mut transposed_chunk = [array_to_m512([0.0; 16]); 16];

        for i in 0..16 {
            transposed_chunk[i] = array_to_m512([
                split_chunk[0][i],
                split_chunk[1][i],
                split_chunk[2][i],
                split_chunk[3][i],
                split_chunk[4][i],
                split_chunk[5][i],
                split_chunk[6][i],
                split_chunk[7][i],
                split_chunk[8][i],
                split_chunk[9][i],
                split_chunk[10][i],
                split_chunk[11][i],
                split_chunk[12][i],
                split_chunk[13][i],
                split_chunk[14][i],
                split_chunk[15][i],
            ]);
        }

        transposed_chunk
    }

    fn get_padded_chunk(array: &Array<2>, row: usize, column: usize) -> [__m512; 16] {
        let mut padded_chunk = [array_to_m512([0.0; 16]); 16];
        let column_chunks = array.shape[1].div_ceil(16);
        let row_start = row * 16;
        let row_end = ((row + 1) * 16).min(array.shape[0]);

        for i in 0..(row_end - row_start) {
            padded_chunk[i] = array.data[(i + row_start) * column_chunks + column];
        }

        padded_chunk
    }

    pub fn matrix_multiplication(&self, matrix_b: &Self) -> Self {
        let matrix_a = self;
        let column_chunks_a = matrix_a.shape[1].div_ceil(16);
        let row_chunks_b = matrix_b.shape[0].div_ceil(16);
        let column_chunks_b = matrix_b.shape[1].div_ceil(16);
        let mut result_data = Vec::with_capacity(matrix_a.shape[0] * column_chunks_b);

        unsafe {
            for row_a in 0..matrix_a.shape[0] {
                for chunk_column_b in 0..column_chunks_b {
                    let mut temp_results = [array_to_m512([0.0; 16]); 16];

                    for chunk_inner_loop_index in 0..row_chunks_b { // TODO: change loop order to do transposing only once per chunk
                        let chunk_b = Self::get_padded_chunk(matrix_b, chunk_inner_loop_index, chunk_column_b);
                        let transposed_chunk_b = Self::transpose_chunk(&chunk_b);
                        let register_a = matrix_a.data[row_a * column_chunks_a + chunk_inner_loop_index];
                        
                        for i in 0..16 {
                            temp_results[i] = _mm512_fmadd_ps(register_a, transposed_chunk_b[i], temp_results[i]);
                        }
                    }

                    let mut result_register = [0.0; 16];

                    for i in 0..16 {
                        result_register[i] = _mm512_reduce_add_ps(temp_results[i]);
                    }

                    result_data.push(array_to_m512(result_register));
                }
            }
        }

        Self {
            data: result_data,
            shape: [matrix_a.shape[0], matrix_b.shape[1]]
        }
    }
}

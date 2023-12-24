use std::arch::x86_64::{_mm512_add_ps, _mm512_mask_add_ps, _mm512_mul_ps, _mm512_mask_mul_ps, _mm512_reduce_add_ps, _mm512_reduce_mul_ps};

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
}

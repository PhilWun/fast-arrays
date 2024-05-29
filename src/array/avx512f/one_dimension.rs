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

use std::arch::x86_64::{
    __m512, _mm512_add_ps, _mm512_broadcastss_ps, _mm512_castps512_ps128, _mm512_mask_add_ps,
    _mm512_mask_max_ps, _mm512_mask_min_ps, _mm512_mask_mul_ps, _mm512_max_ps, _mm512_min_ps,
    _mm512_mul_ps, _mm512_permutexvar_ps, _mm512_reduce_add_ps, _mm512_reduce_max_ps,
    _mm512_reduce_min_ps, _mm512_reduce_mul_ps,
};

use crate::{array::avx512f::array_to_m512i, Array};

use super::{array_to_m512, assert_same_shape2, m512_to_array, reduce};

impl From<Array<1>> for Vec<f32> {
    fn from(value: Array<1>) -> Self {
        let mut converted = vec![0f32; value.shape[0]];
        let mut index: usize = 0;

        for register in value.data {
            let register = m512_to_array(register);

            for i in 0..16 {
                if index >= value.shape[0] {
                    break;
                }

                converted[index] = register[i];
                index += 1;
            }
        }

        converted
    }
}

impl From<Vec<f32>> for Array<1> {
    fn from(value: Vec<f32>) -> Self {
        let register_count = value.len().div_ceil(16);
        let mut data: Vec<__m512> = Vec::with_capacity(register_count);
        let mut index = 0;

        for _ in 0..register_count {
            let mut new_register_data = [0f32; 16];

            for i in 0..16 {
                if index < value.len() {
                    new_register_data[i] = value[index];
                    index += 1;
                }
            }

            data.push(array_to_m512(new_register_data));
        }

        Array {
            data: data,
            shape: [value.len()],
        }
    }
}

impl Array<1> {
    pub fn get(&self, index: usize) -> f32 {
        if index >= self.shape[0] {
            panic!(
                "tried to get index {}, but the array has only {} element(s)",
                index, self.shape[0]
            );
        }

        let register_index = index / 16;
        let value_index = index % 16;

        let value = m512_to_array(self.data[register_index])[value_index];

        value
    }

    pub fn set(&mut self, index: usize, value: f32) {
        if index >= self.shape[0] {
            panic!(
                "tried to set index {}, but the array has only {} element(s)",
                index, self.shape[0]
            );
        }

        let register_index = index / 16;
        let value_index = index % 16;

        let mut new_register = m512_to_array(self.data[register_index]);
        new_register[value_index] = value;

        self.data[register_index] = array_to_m512(new_register);
    }

    pub fn sum(&self) -> f32 {
        if self.shape[0] == 0 {
            return 0.0;
        }

        unsafe {
            let sum_register = reduce(
                &self.data,
                self.shape[0],
                0.0,
                _mm512_add_ps,
                _mm512_mask_add_ps,
            );
            _mm512_reduce_add_ps(sum_register)
        }
    }

    pub fn product(&self) -> f32 {
        if self.shape[0] == 0 {
            return 1.0;
        }

        unsafe {
            let product_register = reduce(
                &self.data,
                self.shape[0],
                1.0,
                _mm512_mul_ps,
                _mm512_mask_mul_ps,
            );
            _mm512_reduce_mul_ps(product_register)
        }
    }

    pub fn max_reduce(&self) -> f32 {
        if self.shape[0] == 0 {
            return f32::MIN;
        }

        unsafe {
            let max_register = reduce(
                &self.data,
                self.shape[0],
                f32::MIN,
                _mm512_max_ps,
                _mm512_mask_max_ps,
            );
            _mm512_reduce_max_ps(max_register)
        }
    }

    pub fn min_reduce(&self) -> f32 {
        if self.shape[0] == 0 {
            return f32::MAX;
        }

        unsafe {
            let min_register = reduce(
                &self.data,
                self.shape[0],
                f32::MAX,
                _mm512_min_ps,
                _mm512_mask_min_ps,
            );
            _mm512_reduce_min_ps(min_register)
        }
    }

    pub fn dot_product(&self, other: &Self) -> f32 {
        assert_same_shape2(self, other);

        if self.shape[0] == 0 {
            return 0.0;
        }

        let mut sum_register = array_to_m512([0.0; 16]);
        let mut last_register_mask = 0xFFFF;

        if self.shape[0] % 16 != 0 {
            last_register_mask = 0xFFFF >> (16 - (self.shape[0] % 16));
        }

        unsafe {
            for (d1, d2) in self.data[0..self.data.len() - 1]
                .iter()
                .zip(other.data[0..other.data.len() - 1].iter())
            {
                sum_register = _mm512_add_ps(sum_register, _mm512_mul_ps(*d1, *d2));
            }

            sum_register = _mm512_mask_add_ps(
                sum_register,
                last_register_mask,
                sum_register,
                _mm512_mul_ps(*self.data.last().unwrap(), *other.data.last().unwrap()),
            );

            _mm512_reduce_add_ps(sum_register)
        }
    }

    /// Copy the array `k`-times into `output`
    pub fn tile_in_place(&self, k: usize, output: &mut Array<1>) {
        assert!(
            self.shape[0] % 16 == 0,
            "the number of elements needs to be a multiple of 16"
        );
        assert_eq!(
            self.shape[0] * k,
            output.shape[0],
            "the number of elements in output must be k-times more than the elements in this array"
        );

        let self_registers = self.data.len();

        for (i, d) in output.data.iter_mut().enumerate() {
            *d = self.data[i % self_registers];
        }
    }

    /// Repeat each element of the array `k`-times and store the result in `output`
    pub fn repeat_in_place(&self, k: usize, output: &mut Array<1>) {
        let self_len = self.shape[0];

        assert!(
            self_len % 16 == 0,
            "the number of elements needs to be a multiple of 16"
        );
        assert!(k % 16 == 0, "k needs to be a multiple of 16");
        assert_eq!(
            self_len * k,
            output.shape[0],
            "the number of elements in output must be k-times more than the elements in this array"
        );

        let permutation_indices =
            array_to_m512i([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]);
        let mut index = 0;

        unsafe {
            for i in 0..self.data.len() {
                let mut register = self.data[i];

                for _ in 0..16 {
                    for _ in 0..k / 16 {
                        output.data[index] =
                            _mm512_broadcastss_ps(_mm512_castps512_ps128(register));
                        index += 1;
                    }

                    register = _mm512_permutexvar_ps(permutation_indices, register);
                }
            }
        }
    }

    pub fn repeat_as_row_in_place(&self, k: usize, output: &mut Array<2>) {
        assert_eq!(output.shape[0], k);
        assert_eq!(output.shape[1], self.shape[0]);

        let registers_per_row = self.shape[0].div_ceil(16);

        for (i, m) in output.data.iter_mut().enumerate() {
            *m = self.data[i % registers_per_row];
        }
    }

    pub fn repeat_as_column_in_place(&self, k: usize, output: &mut Array<2>) {
        assert_eq!(output.shape[0], self.shape[0]);
        assert_eq!(output.shape[1], k);

        let registers_per_row = k.div_ceil(16);

        for i in 0..output.shape[0] {
            let register = array_to_m512([self.get(i); 16]);

            for j in 0..registers_per_row {
                output.data[i * registers_per_row + j] = register;
            }
        }
    }
}

/**
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

use std::arch::x86_64::{__m512, _mm512_add_ps, _mm512_mask_add_ps, _mm512_reduce_add_ps, _mm512_mul_ps, _mm512_mask_mul_ps, _mm512_reduce_mul_ps};

use crate::Array;

use super::{m512_to_array, array_to_m512, assert_same_shape2, reduce};

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
            panic!("tried to get index {}, but the array has only {} element(s)", index, self.shape[0]);
        }

        let register_index = index / 16;
        let value_index = index % 16;

        let value = m512_to_array(self.data[register_index])[value_index];

        value
    }

    pub fn set(&mut self, index: usize, value: f32) {
        if index >= self.shape[0] {
            panic!("tried to set index {}, but the array has only {} element(s)", index, self.shape[0]);
        }

        let register_index = index / 16;
        let value_index = index % 16;

        let mut new_register = m512_to_array(self.data[register_index]);
        new_register[value_index] = value;

        self.data[register_index] = array_to_m512(new_register);
    }

    pub fn set_all(&mut self, value: f32) {
        let new_register = array_to_m512([value; 16]);

        for d in self.data.iter_mut() {
            *d = new_register;
        }
    }

    pub fn sum(&self) -> f32 {
        if self.shape[0] == 0 {
            return 0.0;
        }

        unsafe {
            let sum_register = reduce(&self.data, self.shape[0], 0.0, _mm512_add_ps, _mm512_mask_add_ps);
            _mm512_reduce_add_ps(sum_register)
        }
    }

    pub fn product(&self) -> f32 {
        if self.shape[0] == 0 {
            return 1.0;
        }

        unsafe {
            let sum_register = reduce(&self.data, self.shape[0], 1.0, _mm512_mul_ps, _mm512_mask_mul_ps);
            _mm512_reduce_mul_ps(sum_register)
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
            for (d1, d2) in self.data[0..self.data.len() - 1].iter().zip(other.data[0..other.data.len() - 1].iter()) {
                sum_register = _mm512_add_ps(sum_register, _mm512_mul_ps(*d1, *d2));
            }

            sum_register = _mm512_mask_add_ps(sum_register, last_register_mask, sum_register, _mm512_mul_ps(*self.data.last().unwrap(), *other.data.last().unwrap()));

            _mm512_reduce_add_ps(sum_register)
        }
    }
}

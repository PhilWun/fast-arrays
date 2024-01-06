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

use std::arch::x86_64::__mmask16;

use crate::Mask;

impl From<Mask<1>> for Vec<bool> {
    fn from(value: Mask<1>) -> Self {
        let mut converted = vec![false; value.shape[0]];
        let mut index: usize = 0;

        for register in value.masks {
            for i in 0..16 {
                if index >= value.shape[0] {
                    break;
                }

                converted[index] = register & (1 << i) > 0;
                index += 1;
            }
        }

        converted
    }
}

impl From<Vec<bool>> for Mask<1> {
    fn from(value: Vec<bool>) -> Self {
        let register_count = value.len().div_ceil(16);
        let mut masks: Vec<__mmask16> = Vec::with_capacity(register_count);
        let mut index = 0;

        for _ in 0..register_count {
            let mut new_register_data: __mmask16 = 0;

            for i in 0..16 {
                if index < value.len() {
                    new_register_data |= (value[index] as __mmask16) << i;
                    index += 1;
                }
            }

            masks.push(new_register_data);
        }

        Self {
            masks,
            shape: [value.len()],
        }
    }
}

impl Mask<1> {
    pub fn new(len: usize) -> Self {
        let mask_count = len.div_ceil(16);
        let masks = vec![0u16; mask_count];

        Self {
            masks,
            shape: [len]
        }
    }

    pub fn get(&self, index: usize) -> bool {
        let mask = self.masks[index / 16];

        mask & (1 << (index % 16)) > 0
    }
}


impl<const D: usize> Mask<D> {
    pub fn and(&self, other: &Self) -> Self {
        let mut clone = self.clone();
        clone.and_in_place(other);

        clone
    }

    pub fn and_in_place(&mut self, other: &Self) {
        for (m1, m2) in self.masks.iter_mut().zip(other.masks.iter()) {
            *m1 = *m1 & *m2;
        }
    }

    pub fn or(&self, other: &Self) -> Self {
        let mut clone = self.clone();
        clone.or_in_place(other);

        clone
    }

    pub fn or_in_place(&mut self, other: &Self) {
        for (m1, m2) in self.masks.iter_mut().zip(other.masks.iter()) {
            *m1 = *m1 | *m2;
        }
    }

    pub fn not(&self) -> Self {
        let mut clone = self.clone();
        clone.not_in_place();

        clone
    }

    pub fn not_in_place(&mut self) {
        for m in self.masks.iter_mut() {
            *m = !*m;
        }
    }
}

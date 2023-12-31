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

use crate::Mask;

impl From<Mask<1>> for Vec<bool> {
    fn from(value: Mask<1>) -> Self {
        value.masks
    }
}

impl From<Vec<bool>> for Mask<1> {
    fn from(value: Vec<bool>) -> Self {
        let len = value.len();
        Self {
            masks: value,
            shape: [len],
        }
    }
}

impl Mask<1> {
    pub fn new(len: usize) -> Self {
        Self {
            masks: vec![false; len],
            shape: [len]
        }
    }

    pub fn get(&self, index: usize) -> bool {
        self.masks[index]
    }

    /// Copy the mask `k`-times into `output`
    pub fn tile_in_place(&self, k: usize, output: &mut Mask<1>) {
        assert_eq!(self.shape[0] * k, output.shape[0], "the number of elements in output must be k-times more than the elements in this mask");
        let self_masks = self.masks.len();

        for (i, d) in output.masks.iter_mut().enumerate() {
            *d = self.masks[i % self_masks];
        }
    }

    /// Repeat each element of the mask `k`-times and store the result in `output`
    pub fn repeat_in_place(&self, k: usize, output: &mut Mask<1>) {
        assert_eq!(self.shape[0] * k, output.shape[0], "the number of elements in output must be k-times more than the elements in this array");
        let mut index = 0;

        for i in 0..self.masks.len() {
            let mask = self.masks[i];

            for _ in 0..k {
                output.masks[index] = mask;
                index += 1;
            }
        }
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

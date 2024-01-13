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

use std::slice::IterMut;

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

    pub fn repeat_as_row_in_place(&self, k: usize, output: &mut Mask<2>) {
        assert_eq!(output.shape[0], k);
        assert_eq!(output.shape[1], self.shape[0]);

        for (i, m) in output.masks.iter_mut().enumerate() {
            *m = self.masks[i % self.shape[0]];
        }
    }

    pub fn repeat_as_column_in_place(&self, k: usize, output: &mut Mask<2>) {
        assert_eq!(output.shape[0], self.shape[0]);
        assert_eq!(output.shape[1], k);

        for i in 0..output.shape[0] {
            let mask = self.get(i);

            for j in 0..k {
                output.masks[i * k + j] = mask;
            }
        }
    }
}

impl Mask<2> {
    pub fn get(&self, row: usize, column: usize) -> bool {
        self.masks[row * self.shape[1] + column]
    }
}

/// This struct is used to create a mutable iterator over the masks and automatically zero out unused elements afterwards.
pub struct MutableMasks<'a, const D: usize> {
    mask: &'a mut Mask<D>
}

impl<'a, const D: usize> MutableMasks<'a, D> {
    pub fn iter_mut(&mut self) -> IterMut<bool> {
        self.mask.masks.iter_mut()
    }
}

impl<const D: usize> Mask<D> {
    pub fn zeros(shape: &[usize; D]) -> Self {
        let mut n_masks = 1;

        for i in 0..D {
            n_masks *= shape[i];
        }

        Self {
            masks: vec![false; n_masks],
            shape: *shape
        }
    }

    pub(crate) fn new_from_data(shape: [usize; D], masks: Vec<bool>) -> Mask<D> {
        let mut n_masks = 1;

        for i in 0..D {
            n_masks *= shape[i];
        }
        
        assert_eq!(n_masks, masks.len(), "length of masks does not equal the expected length");

        Mask { masks: masks, shape: shape }
    }

    pub fn get_shape(&self) -> &[usize; D] {
        &self.shape
    }

    pub(crate) fn get_masks(&self) -> &Vec<bool> {
        &self.masks
    }

    pub fn get_masks_mut(&mut self) -> MutableMasks<D> {
        MutableMasks {
            mask: self
        }
    }

    pub fn assert_invariants_satisfied(&self) {

    }

    pub fn and(&self, other: &Self) -> Self {
        let mut clone = self.clone();
        clone.and_in_place(other);

        clone
    }

    pub fn and_in_place(&mut self, other: &Self) {
        assert_eq!(self.shape, other.shape);

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
        assert_eq!(self.shape, other.shape);
        
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

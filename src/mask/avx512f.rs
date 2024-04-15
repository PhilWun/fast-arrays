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

use serde::{
    ser::{Serialize, SerializeSeq, SerializeStruct},
    Deserialize,
};
use std::{arch::x86_64::__mmask16, slice::IterMut};

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

struct DataSerializeWrapper<'a, const D: usize>(&'a Mask<D>);

impl<'a, const D: usize> Serialize for DataSerializeWrapper<'a, D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.0.number_of_elements()))?;
        let registers_per_row = self.0.shape.last().unwrap().div_ceil(16);

        for (i, register) in self.0.masks.iter().enumerate() {
            let mut limit = 16;

            // checks if current register is the last register in its row
            if ((i + 1) % registers_per_row) == 0 {
                limit = ((self.0.shape.last().unwrap() - 1) % 16) + 1;
            }

            for i in 0..limit {
                seq.serialize_element(&((register & (1 << i)) > 0))?;
            }
        }

        seq.end()
    }
}

impl<const D: usize> Serialize for Mask<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Array", 2)?;
        state.serialize_field("masks", &DataSerializeWrapper(self))?;

        let shape_vec: Vec<usize> = self.shape.into();
        state.serialize_field("shape", &shape_vec)?;

        state.end()
    }
}

#[derive(Deserialize)]
struct ArrayDeserializerProxy {
    masks: Vec<bool>,
    shape: Vec<usize>,
}

impl<'de, const D: usize> Deserialize<'de> for Mask<D> {
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let proxy = ArrayDeserializerProxy::deserialize(deserializer)?;
        assert_eq!(proxy.shape.len(), D);

        let mut shape = [0; D];
        shape.copy_from_slice(&proxy.shape[..D]);

        let registers_per_row = shape.last().unwrap().div_ceil(16);
        let mut register_count = registers_per_row;

        for d in shape[0..D - 1].iter() {
            register_count *= d;
        }

        let mut element_index = 0;
        let mut masks = vec![0u16; register_count];

        for register_index in 0..register_count {
            let mut limit = 16;

            // checks if current register is the last register in its row
            if ((register_index + 1) % registers_per_row) == 0 {
                limit = ((shape.last().unwrap() - 1) % 16) + 1;
            }

            let content = &proxy.masks[element_index..element_index + limit];
            element_index += limit;

            let mut new_mask = 0u16;

            for (i, mask) in content.iter().enumerate() {
                if *mask {
                    new_mask |= 1 << i;
                }
            }

            masks[register_index] = new_mask;
        }

        Ok(Mask { masks, shape })
    }
}

impl Mask<1> {
    pub fn get(&self, index: usize) -> bool {
        let mask = self.masks[index / 16];

        mask & (1 << (index % 16)) > 0
    }

    /// Copy the mask `k`-times into `output`
    pub fn tile_in_place(&self, k: usize, output: &mut Mask<1>) {
        assert!(
            self.shape[0] % 16 == 0,
            "the number of elements needs to be a multiple of 16"
        );
        assert_eq!(
            self.shape[0] * k,
            output.shape[0],
            "the number of elements in output must be k-times more than the elements in this mask"
        );

        let self_masks = self.masks.len();

        for (i, d) in output.masks.iter_mut().enumerate() {
            *d = self.masks[i % self_masks];
        }
    }

    /// Repeat each element of the mask `k`-times and store the result in `output`
    pub fn repeat_in_place(&self, k: usize, output: &mut Mask<1>) {
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

        let mut index = 0;

        for i in 0..self.masks.len() {
            let mask = self.masks[i];

            for j in 0..16 {
                let new_mask = ((mask >> j) & 1) * 0xFFFF;

                for _ in 0..k / 16 {
                    output.masks[index] = new_mask;
                    index += 1;
                }
            }
        }
    }

    pub fn repeat_as_row_in_place(&self, k: usize, output: &mut Mask<2>) {
        assert_eq!(output.shape[0], k);
        assert_eq!(output.shape[1], self.shape[0]);

        let masks_per_row = self.shape[0].div_ceil(16);

        for (i, m) in output.masks.iter_mut().enumerate() {
            *m = self.masks[i % masks_per_row];
        }
    }

    pub fn repeat_as_column_in_place(&self, k: usize, output: &mut Mask<2>) {
        assert_eq!(output.shape[0], self.shape[0]);
        assert_eq!(output.shape[1], k);

        let masks_per_row = k.div_ceil(16);

        for i in 0..output.shape[0] {
            let mask = self.get(i) as __mmask16 * 0xFFFF;

            for j in 0..masks_per_row {
                output.masks[i * masks_per_row + j] = mask;
            }
        }

        output.zero_out_unused_elements();
    }
}

impl Mask<2> {
    pub fn from_vec(data: &Vec<bool>, shape: [usize; 2]) -> Self {
        assert!(shape[0] > 0);
        assert!(shape[1] > 0);
        assert_eq!(data.len(), shape[0] * shape[1]);

        let row_count = shape[0];
        let column_count = shape[1];
        let masks_per_row = column_count.div_ceil(16);
        let mut new_masks = Vec::with_capacity(masks_per_row * row_count);
        let mut index = 0;

        for _ in 0..row_count {
            for c in 0..masks_per_row {
                let mut mask = 0;
                let mut limit = 16;

                // if it is the last mask in the row
                if (c + 1) % masks_per_row == 0 {
                    limit = (column_count - 1) % 16 + 1;
                }

                for i in 0..limit {
                    mask |= (data[index] as u16) << i;
                    index += 1;
                }

                new_masks.push(mask);
            }
        }

        Self {
            masks: new_masks,
            shape,
        }
    }

    pub fn get(&self, row: usize, column: usize) -> bool {
        let masks_per_row = self.shape[1].div_ceil(16);

        self.masks[row * masks_per_row + (column / 16)] & (1 << (column % 16)) > 0
    }
}

/// This struct is used to create a mutable iterator over the masks and automatically zero out unused elements afterwards.
pub struct MutableMasks<'a, const D: usize> {
    mask: &'a mut Mask<D>,
}

impl<'a, const D: usize> MutableMasks<'a, D> {
    pub fn iter_mut(&mut self) -> IterMut<__mmask16> {
        self.mask.masks.iter_mut()
    }
}

impl<'a, const D: usize> Drop for MutableMasks<'a, D> {
    fn drop(&mut self) {
        self.mask.zero_out_unused_elements();
    }
}

impl<const D: usize> Mask<D> {
    pub fn zeros(shape: &[usize; D]) -> Self {
        let mut mask_count = shape.last().unwrap().div_ceil(16);

        for i in 0..D - 1 {
            mask_count *= shape[i];
        }

        let masks = vec![0u16; mask_count];

        Self {
            masks,
            shape: *shape,
        }
    }

    pub fn new_from_data(shape: [usize; D], masks: Vec<__mmask16>) -> Mask<D> {
        let masks_per_row = shape.last().unwrap().div_ceil(16);
        let mut n_masks = masks_per_row;

        for i in 0..D - 1 {
            n_masks *= shape[i];
        }

        assert_eq!(
            n_masks,
            masks.len(),
            "length of masks does not equal the expected length"
        );

        let mut new_mask = Mask {
            masks: masks,
            shape: shape,
        };
        new_mask.zero_out_unused_elements();

        new_mask
    }

    pub fn get_shape(&self) -> &[usize; D] {
        &self.shape
    }

    pub(crate) fn get_masks(&self) -> &Vec<__mmask16> {
        &self.masks
    }

    pub fn assert_invariants_satisfied(&self) {
        // check number of masks
        let masks_per_row = self.shape.last().unwrap().div_ceil(16);
        let mut n_masks = masks_per_row;

        for i in 0..D - 1 {
            n_masks *= self.shape[i];
        }

        assert_eq!(
            self.masks.len(),
            n_masks,
            "number of masks does not match the expected number"
        );

        // check that unused bits are 0
        if self.shape.last().unwrap() % 16 == 0 {
            return;
        }

        let masks_per_row = self.shape.last().unwrap().div_ceil(16);
        let unused_elements_mask = 0xFFFF << (self.shape.last().unwrap() % 16);
        let mut rows = 1;

        for s in 0..D - 1 {
            rows *= self.shape[s];
        }

        for r in 1..rows + 1 {
            assert_eq!(
                self.masks[r * masks_per_row - 1] & unused_elements_mask,
                0,
                "mask contains unused bits that are set to 1"
            );
        }
    }

    /// set everything outside the bounds of the shape to zero
    pub(crate) fn zero_out_unused_elements(&mut self) {
        let out_of_bound_elements = 16 - (self.shape.last().unwrap() % 16);

        if out_of_bound_elements == 16 {
            return;
        }

        let out_of_bound_mask = 0xFFFF >> out_of_bound_elements;
        let mut rows = 1;
        let masks_per_row = self.shape.last().unwrap().div_ceil(16);

        for s in 0..D - 1 {
            rows *= self.shape[s];
        }

        for r in 1..rows + 1 {
            self.masks[r * masks_per_row - 1] &= out_of_bound_mask;
        }
    }

    pub fn get_masks_mut(&mut self) -> MutableMasks<D> {
        MutableMasks { mask: self }
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

        self.zero_out_unused_elements();
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    #[allow(unused_imports)]
    use super::Mask;

    #[rstest]
    #[should_panic]
    fn unused_elements_are_zero_1d(
        #[values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17)] rows: usize,
    ) {
        let n_masks = rows.div_ceil(16);
        let masks = vec![0xFFFF; n_masks];

        let mask = Mask {
            masks,
            shape: [rows],
        };

        mask.assert_invariants_satisfied();
    }

    #[rstest]
    #[should_panic]
    fn number_of_elements_1d(
        #[values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33)] rows: usize,
    ) {
        let masks = vec![0; 2];

        let mask = Mask {
            masks,
            shape: [rows],
        };

        mask.assert_invariants_satisfied();
    }

    #[rstest]
    #[should_panic]
    fn unused_elements_are_zero_2d(
        #[values(1, 2, 3)] columns: usize,
        #[values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17)] rows: usize,
    ) {
        let n_masks = rows.div_ceil(16) * columns;
        let masks = vec![0xFFFF; n_masks];

        let mask = Mask {
            masks,
            shape: [rows],
        };

        mask.assert_invariants_satisfied();
    }

    #[rstest]
    #[should_panic]
    fn number_of_elements_2d(
        #[values(1, 2, 3)] columns: usize,
        #[values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33)] rows: usize,
    ) {
        let masks = vec![0; 2 * columns];

        let mask = Mask {
            masks,
            shape: [rows],
        };

        mask.assert_invariants_satisfied();
    }

    #[test]
    fn zero_out_unused_elements_1d() {
        for i in 1..64usize {
            let n_masks = i.div_ceil(16);
            let masks = vec![0xFFFF; n_masks];
            let shape = [i];
            let mut mask = Mask { masks, shape };

            mask.zero_out_unused_elements();

            for j in 0..n_masks * 16 {
                let expected_bit = if j < i { 1 } else { 0 };

                let actual_bit = (mask.masks[j / 16] >> (j % 16)) & 1;

                assert_eq!(expected_bit, actual_bit);
            }
        }
    }

    #[test]
    fn zero_out_unused_elements_2d() {
        for i in 1..64usize {
            for j in 1..64usize {
                let masks_per_row = j.div_ceil(16);
                let masks = vec![0xFFFF; i * masks_per_row];
                let shape = [i, j];
                let mut mask = Mask { masks, shape };

                mask.zero_out_unused_elements();

                for k in 0..i {
                    for l in 0..masks_per_row * 16 {
                        let expected_bit = if l < j { 1 } else { 0 };

                        let actual_bit = (mask.masks[k * masks_per_row + (l / 16)] >> (l % 16)) & 1;

                        assert_eq!(expected_bit, actual_bit);
                    }
                }
            }
        }
    }

    #[test]
    fn get_masks_mut_1d() {
        for i in 1..64usize {
            let n_masks = i.div_ceil(16);
            let masks = vec![0xFFFF; n_masks];
            let shape = [i];
            let mut mask = Mask { masks, shape };

            for m in mask.get_masks_mut().iter_mut() {
                *m = 0xFFFF;
            }

            for j in 0..n_masks * 16 {
                let expected_bit = if j < i { 1 } else { 0 };

                let actual_bit = (mask.masks[j / 16] >> (j % 16)) & 1;

                assert_eq!(expected_bit, actual_bit);
            }
        }
    }

    #[test]
    fn get_masks_mut_2d() {
        for i in 1..64usize {
            for j in 1..64usize {
                let masks_per_row = j.div_ceil(16);
                let masks = vec![0; i * masks_per_row];
                let shape = [i, j];
                let mut mask = Mask { masks, shape };

                for m in mask.get_masks_mut().iter_mut() {
                    *m = 0xFFFF;
                }

                for k in 0..i {
                    for l in 0..masks_per_row * 16 {
                        let expected_bit = if l < j { 1 } else { 0 };

                        let actual_bit = (mask.masks[k * masks_per_row + (l / 16)] >> (l % 16)) & 1;

                        assert_eq!(expected_bit, actual_bit);
                    }
                }
            }
        }
    }
}

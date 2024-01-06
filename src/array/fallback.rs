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

use rand::{distributions::{Uniform, Distribution}, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{Array, Mask};

impl<const D: usize> From<Array<D>> for Vec<f32> {
    fn from(value: Array<D>) -> Self {
        value.data
    }
}

impl From<Vec<f32>> for Array<1> {
    fn from(value: Vec<f32>) -> Self {
        let len = value.len();

        Array {
            data: value,
            shape: [len]
        }
    }
}

fn calculate_size(shape: &[usize]) -> usize {
    let mut size = 1;

    for s in shape {
        size *= s;
    }

    size
}

fn assert_same_shape2<const D: usize>(a: &Array<D>, b: &Array<D>) {
    assert_eq!(a.shape, b.shape, "the lengths of array one and two don't match: {:?} != {:?}", a.shape, b.shape);
}

fn assert_same_shape_with_mask2<const D: usize>(a: &Array<D>, b: &Array<D>, mask: &Mask<D>) {
    assert_eq!(a.shape, b.shape, "the lengths of array one and two don't match: {:?} != {:?}", a.shape, b.shape);
    assert_eq!(a.shape, mask.shape, "the lengths of array one and mask don't match: {:?} != {:?}", a.shape, mask.shape);
}

fn assert_same_shape3<const D: usize>(a: &Array<D>, b: &Array<D>, c: &Array<D>) {
    assert_eq!(a.shape, b.shape, "the lengths of array one and two don't match: {:?} != {:?}", a.shape, b.shape);
    assert_eq!(b.shape, c.shape, "the lengths of array two and three don't match: {:?} != {:?}", b.shape, c.shape);
}

fn assert_same_shape_with_mask3<const D: usize>(a: &Array<D>, b: &Array<D>, c: &Array<D>, mask: &Mask<D>) {
    assert_eq!(a.shape, b.shape, "the lengths of array one and two don't match: {:?} != {:?}", a.shape, b.shape);
    assert_eq!(b.shape, c.shape, "the lengths of array two and three don't match: {:?} != {:?}", b.shape, c.shape);
    assert_eq!(a.shape, mask.shape, "the lengths of array one and mask don't match: {:?} != {:?}", a.shape, mask.shape);
}

impl<const D: usize> Array<D> {
    pub fn zeros(shape: &[usize; D]) -> Self {
        Self {
            data: vec![0.0; calculate_size(shape)],
            shape: *shape,
        }
    }

    pub fn new_from_value(shape: &[usize; D], value: f32) -> Self {
        Self {
            data: vec![value; calculate_size(shape)],
            shape: *shape,
        }
    }

    pub fn random_uniform(shape: &[usize; D], min: f32, max: f32, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(seed) => ChaCha20Rng::seed_from_u64(seed),
            None => ChaCha20Rng::from_entropy(),
        };

        let distribution = Uniform::new(min, max);
        let size = calculate_size(shape);
        let mut data = Vec::with_capacity(size);

        for _ in 0..size {
            data.push(distribution.sample(&mut rng));
        }

        Self {
            data,
            shape: *shape
        }
    }

    /// set the elements to `value` where `mask` is 1
    pub fn set_masked(&mut self, value: f32, mask: &Mask<D>) {
        assert_eq!(self.shape, mask.shape); // TODO: add messages to asserts

        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = value;
            }
        }
    }

    /// set the elements to `v1` where `mask` is 0 and to `v2` where `mask` is 1
    pub fn set_masked2(&mut self, v1: f32, v2: f32, mask: &Mask<D>) {
        assert_eq!(self.shape, mask.shape);

        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = v2;
            } else {
                *d = v1;
            }
        }
    }

    // copy the elements from `other` where `mask` is 1
    pub fn copy_masked(&mut self, other: &Array<D>, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, other, mask);

        for ((d1, d2), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
            if *m {
                *d1 = *d2;
            }
        }
    }

    // copy the elements from `other1` where `mask` is 0 and from `other2` where `mask` is 1
    pub fn copy_masked2(&mut self, other1: &Array<D>, other2: &Array<D>, mask: &Mask<D>) {
        assert_same_shape_with_mask3(&self, other1, other2, mask);

        for (((d1, d2), d3), m) in self.data.iter_mut().zip(other1.data.iter()).zip(other2.data.iter()).zip(mask.masks.iter()) {
            if *m {
                *d1 = *d3;
            } else {
                *d1 = *d2;
            }
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.add_in_place(other);

        new_array
    }

    pub fn add_in_place(&mut self, other: &Self) {
        assert_same_shape2(&self, &other);

        for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
            *l = *l + *r;
        }
    }

    pub fn add_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
            if *m {
                *l = *l + *r;
            }
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.sub_in_place(other);

        new_array
    }

    pub fn sub_in_place(&mut self, other: &Self) {
        assert_same_shape2(&self, &other);

        for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
            *l = *l - *r;
        }
    }

    pub fn sub_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
            if *m {
                *l = *l - *r;
            }
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.mul_in_place(other);

        new_array
    }

    pub fn mul_in_place(&mut self, other: &Self) {
        assert_same_shape2(&self, &other);

        for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
            *l = *l * *r;
        }
    }

    pub fn mul_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
            if *m {
                *l = *l * *r;
            }
        }
    }

    pub fn div(&self, other: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.div_in_place(other);

        new_array
    }

    pub fn div_in_place(&mut self, other: &Self) {
        assert_same_shape2(&self, &other);

        for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
            *l = *l / *r;
        }
    }

    pub fn div_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
            if *m {
                *l = *l / *r;
            }
        }
    }

    pub fn max(&self, other: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.max_in_place(other);

        new_array
    }

    pub fn max_in_place(&mut self, other: &Self) {
        assert_same_shape2(self, other);

        for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
            *l = l.max(*r);
        }
    }

    pub fn max_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
            if *m {
                *l = l.max(*r);
            }
        }
    }

    pub fn min(&self, other: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.min_in_place(other);

        new_array
    }

    pub fn min_in_place(&mut self, other: &Self) {
        assert_same_shape2(self, other);

        for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
            *l = l.min(*r);
        }
    }

    pub fn min_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
            if *m {
                *l = l.min(*r);
            }
        }
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.add_scalar_in_place(scalar);

        new_array
    }

    pub fn add_scalar_in_place(&mut self, scalar: f32) {
        for d in self.data.iter_mut() {
            *d = *d + scalar;
        }
    }

    pub fn add_scalar_in_place_masked(&mut self, scalar: f32, mask: &Mask<D>) {
        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = *d + scalar;
            }
        }
    }

    pub fn sub_scalar(&self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.sub_scalar_in_place(scalar);

        new_array
    }

    pub fn sub_scalar_in_place(&mut self, scalar: f32) {
        for d in self.data.iter_mut() {
            *d = *d - scalar;
        }
    }

    pub fn sub_scalar_in_place_masked(&mut self, scalar: f32, mask: &Mask<D>) {
        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = *d - scalar;
            }
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.mul_scalar_in_place(scalar);

        new_array
    }

    pub fn mul_scalar_in_place(&mut self, scalar: f32) {
        for d in self.data.iter_mut() {
            *d = *d * scalar;
        }
    }

    pub fn mul_scalar_in_place_masked(&mut self, scalar: f32, mask: &Mask<D>) {
        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = *d * scalar;
            }
        }
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.div_scalar_in_place(scalar);

        new_array
    }

    pub fn div_scalar_in_place(&mut self, scalar: f32) {
        for d in self.data.iter_mut() {
            *d = *d / scalar;
        }
    }

    pub fn div_scalar_in_place_masked(&mut self, scalar: f32, mask: &Mask<D>) {
        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = *d / scalar;
            }
        }
    }

    pub fn fmadd(&self, a: &Self, b: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.fmadd_in_place(a, b);

        new_array
    }

    pub fn fmadd_in_place(&mut self, a: &Self, b: &Self)  {
        assert_same_shape3(self, a, b);

        for ((a, b), c) in a.data.iter().zip(b.data.iter()).zip(self.data.iter_mut()) {
            *c = *a * *b + *c;
        }
    }

    pub fn fmadd_in_place_masked(&mut self, a: &Self, b: &Self, mask: &Mask<D>)  {
        assert_same_shape_with_mask3(self, a, b, mask);

        for (((a, b), c), m) in a.data.iter().zip(b.data.iter()).zip(self.data.iter_mut()).zip(mask.masks.iter()) {
            if *m {
                *c = *a * *b + *c;
            }
        }
    }

    pub fn fmadd_scalar(&self, a: &Self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.fmadd_scalar_in_place(a, scalar);

        new_array
    }

    pub fn fmadd_scalar_in_place(&mut self, a: &Self, scalar: f32)  {
        assert_same_shape2(self, a);

        for (a, b) in a.data.iter().zip(self.data.iter_mut()) {
            *b = *a * scalar + *b;
        }
    }

    pub fn fmadd_scalar_in_place_masked(&mut self, a: &Self, scalar: f32, mask: &Mask<D>)  {
        assert_same_shape_with_mask2(self, a, mask);

        for ((a, c), m) in a.data.iter().zip(self.data.iter_mut()).zip(mask.masks.iter()) {
            if *m {
                *c = *a * scalar + *c;
            }
        }
    }

    pub fn sqrt(&self) -> Self {
        let mut new_array = self.clone();
        new_array.sqrt_in_place();

        new_array
    }

    pub fn sqrt_in_place(&mut self) {
        for d in self.data.iter_mut() {
            *d = d.sqrt();
        }
    }

    pub fn sqrt_in_place_masked(&mut self, mask: &Mask<D>) {
        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = d.sqrt();
            }
        }
    }

    pub fn square(&self) -> Self {
        let mut new_array = self.clone();
        new_array.square_in_place();

        new_array
    }

    pub fn square_in_place(&mut self) {
        for d in self.data.iter_mut() {
            *d = *d * *d;
        }
    }

    pub fn square_in_place_masked(&mut self, mask: &Mask<D>) {
        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = *d * *d;
            }
        }
    }

    pub fn abs(&self) -> Self {
        let mut new_array = self.clone();
        new_array.abs_in_place();

        new_array
    }

    pub fn abs_in_place(&mut self) {
        for d in self.data.iter_mut() {
            *d = d.abs();
        }
    }

    pub fn abs_in_place_masked(&mut self, mask: &Mask<D>) {
        for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
            if *m {
                *d = d.abs();
            }
        }
    }

    pub fn sum(&self) -> f32 {
        let mut sum = 0.0;

        for v in self.data.iter() {
            sum += v;
        }

        sum
    }

    pub fn product(&self) -> f32 {
        let mut product = 1.0;

        for v in self.data.iter() {
            product *= v;
        }

        product
    }

    fn compare(a: &Array<D>, b: &Array<D>, func: fn(&f32, &f32) -> bool) -> Mask<D> {
        assert_same_shape2(a, b);
        let mut data = Vec::with_capacity(a.data.len());

        for (d1, d2) in a.data.iter().zip(b.data.iter()) {
            data.push(func(d1, d2));
        }

        Mask {
            masks: data,
            shape: a.shape
        }
    }

    fn compare_in_place(a: &Array<D>, b: &Array<D>, mask: &mut Mask<D>, func: fn(&f32, &f32) -> bool) {
        assert_same_shape_with_mask2(a, b, &mask);

        for ((d1, d2), m) in a.data.iter().zip(b.data.iter()).zip(mask.masks.iter_mut()) {
            *m = func(d1, d2);
        }
    }

    fn compare_scalar(a: &Array<D>, scalar: f32, func: fn(&f32, &f32) -> bool) -> Mask<D> {
        let mut data = Vec::with_capacity(a.data.len());

        for d in a.data.iter() {
            data.push(func(d, &scalar));
        }

        Mask {
            masks: data,
            shape: a.shape
        }
    }

    fn compare_scalar_in_place(a: &Array<D>, scalar: f32, mask: &mut Mask<D>, func: fn(&f32, &f32) -> bool) {
        assert_eq!(a.shape, mask.shape);
        
        for (d, m) in a.data.iter().zip(mask.masks.iter_mut()) {
            *m = func(d, &scalar);
        }
    }

    pub fn compare_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, f32::eq)
    }

    pub fn compare_equal_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, f32::eq)
    }

    pub fn compare_scalar_equal(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, f32::eq)
    }

    pub fn compare_scalar_equal_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, f32::eq)
    }

    pub fn compare_not_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, f32::ne)
    }

    pub fn compare_not_equal_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, f32::ne)
    }

    pub fn compare_scalar_not_equal(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, f32::ne)
    }

    pub fn compare_scalar_not_equal_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, f32::ne)
    }

    pub fn compare_greater_than(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, f32::gt)
    }

    pub fn compare_greater_than_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, f32::gt)
    }

    pub fn compare_scalar_greater_than(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, f32::gt)
    }

    pub fn compare_scalar_greater_than_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, f32::gt)
    }

    pub fn compare_greater_than_or_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, f32::ge)
    }

    pub fn compare_greater_than_or_equal_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, f32::ge)
    }

    pub fn compare_scalar_greater_than_or_equal(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, f32::ge)
    }

    pub fn compare_scalar_greater_than_or_equal_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, f32::ge)
    }

    pub fn compare_less_than(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, f32::lt)
    }

    pub fn compare_less_than_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, f32::lt)
    }

    pub fn compare_scalar_less_than(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, f32::lt)
    }

    pub fn compare_scalar_less_than_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, f32::lt)
    }

    pub fn compare_less_than_or_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, f32::le)
    }

    pub fn compare_less_than_or_equal_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, f32::le)
    }

    pub fn compare_scalar_less_than_or_equal(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, f32::le)
    }

    pub fn compare_scalar_less_than_or_equal_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, f32::le)
    }

    pub fn exp(&self) -> Self {
        let mut tmp = self.clone();
        tmp.exp_in_place();

        tmp
    }

    pub fn exp_in_place(&mut self) {
        for d in self.data.iter_mut() {
            *d = d.exp()
        }
    }
}

impl Array<1> {
    pub fn get(&self, index: usize) -> f32 {
        if index >= self.shape[0] {
            panic!("tried to get index {}, but the array has only {} element(s)", index, self.shape[0]);
        }

        self.data[index]
    }

    pub fn set(&mut self, index: usize, value: f32) {
        if index >= self.shape[0] {
            panic!("tried to set index {}, but the array has only {} element(s)", index, self.shape[0]);
        }

        self.data[index] = value;
    }

    pub fn set_all(&mut self, value: f32) {
        for v in self.data.iter_mut() {
            *v = value;
        }
    }

    pub fn dot_product(&self, other: &Self) -> f32 {
        let mut result = 0.0;

        for (v1, v2) in self.data.iter().zip(other.data.iter()) {
            result += v1 * v2;
        }

        result
    }
}

impl Array<2> {
    pub fn from_vec(data: &Vec<f32>, shape: [usize; 2]) -> Self {
        assert!(shape[0] > 0);
        assert!(shape[1] > 0);

        Self {
            data: data.clone(),
            shape
        }
    }

    pub fn vector_multiplication(&self, other: &Array<1>) -> Array<1> {
        let rows = self.shape[0];
        let columns = self.shape[1];
        let mut result = Vec::with_capacity(columns);

        for i in 0..rows {
            let mut sum = 0.0;

            for j in 0..columns {
                sum += self.data[i * columns + j] * other.data[j];
            }

            result.push(sum);
        }

        Array::<1> {
            data: result,
            shape: [rows],
        }
    }

    pub fn matrix_multiplication(&self, matrix_b: &Self) -> Self {
        let a_rows = self.shape[0];
        let a_columns = self.shape[1];
        let b_columns = matrix_b.shape[1];
        let mut result = vec![0.0; a_rows * b_columns];

        for a_row in 0..a_rows {
            for b_column in 0..b_columns {
                for inner_loop_index in 0..a_columns {
                    result[a_row * b_columns + b_column] += self.data[a_row * a_columns + inner_loop_index] * matrix_b.data[inner_loop_index * b_columns + b_column];
                }
            }
        }

        Self {
            data: result,
            shape: [a_rows, b_columns],
        }
    }
}

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

mod one_dimension;
mod two_dimensions;

use std::{
    arch::x86_64::{__m512, _mm512_add_ps, _mm512_sub_ps, _mm512_mul_ps, _mm512_div_ps, _mm512_max_ps, _mm512_min_ps, _mm512_sqrt_ps, _mm512_fmadd_ps, _mm512_abs_ps, _mm512_cmpeq_ps_mask, _mm512_cmpneq_ps_mask, _mm512_cmpnle_ps_mask, _mm512_cmpnlt_ps_mask, _mm512_cmplt_ps_mask, _mm512_cmple_ps_mask, _mm512_mul_round_ps, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_NO_EXC, _mm512_cvtps_epi32, _mm512_slli_epi32, _mm512_castsi512_ps, _mm512_add_epi32, _mm512_castps_si512, __mmask16, _mm512_mask_add_ps, _mm512_mask_sub_ps, _mm512_mask_mul_ps, _mm512_mask_div_ps, _mm512_mask_max_ps, _mm512_mask_min_ps, _mm512_mask3_fmadd_ps, _mm512_mask_sqrt_ps, _mm512_mask_abs_ps},
    simd::f32x16
};

use rand::{distributions::{Uniform, Distribution}, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{Array, Mask};

fn m512_to_array(value: __m512) -> [f32; 16] {
    let value: f32x16 = value.into();
    value.into()
}

fn array_to_m512(value: [f32; 16]) -> __m512 {
    let value: f32x16 = value.into();
    value.into()
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

unsafe fn reduce(data: &[__m512], len: usize, default_value: f32, func: unsafe fn(__m512, __m512) -> __m512, mask_func: unsafe fn(__m512, __mmask16, __m512, __m512) -> __m512) -> __m512 {
    let mut result_register = array_to_m512([default_value; 16]);
    let mut last_register_mask = 0xFFFF;

    if len % 16 != 0 {
        last_register_mask = 0xFFFF >> (16 - (len % 16));
    }

    unsafe {
        for d in data[0..data.len() - 1].iter() {
            result_register = func(result_register, *d);
        }

        result_register = mask_func(result_register, last_register_mask, result_register, *data.last().unwrap());
    }

    result_register
}

fn calculate_register_count(shape: &[usize]) -> usize {
    let mut register_count = shape.last().unwrap().div_ceil(16);

    for i in 0..shape.len() - 1 {
        register_count *= shape[i];
    }

    register_count
}

impl<const D: usize> Array<D> {
    pub fn zeros(shape: &[usize; D]) -> Self {
        Self::new_from_value(shape, 0.0)
    }

    pub fn new_from_value(shape: &[usize; D], value: f32) -> Self {
        assert!(D > 0);
        
        let register_count = calculate_register_count(shape);
        let zero = array_to_m512([value; 16]);
        let data = vec![zero; register_count];

        Self {
            data,
            shape: *shape,
        }
    }

    pub fn random_uniform(shape: &[usize; D], min: f32, max: f32, seed: Option<u64>) -> Self {
        let mut new_array = Self::zeros(shape);
        new_array.random_uniform_in_place(min, max, seed);

        new_array
    }

    pub fn random_uniform_in_place(&mut self, min: f32, max: f32, seed: Option<u64>) {
        let mut rng = match seed {
            Some(seed) => ChaCha20Rng::seed_from_u64(seed),
            None => ChaCha20Rng::from_entropy(),
        };

        let distribution = Uniform::new(min, max);
        let mut tmp_register_data = [0.0; 16];

        for d in self.data.iter_mut() {
            for i in 0..16 {
                tmp_register_data[i] = distribution.sample(&mut rng);
            }

            *d = array_to_m512(tmp_register_data);
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.add_in_place(other);

        new_array
    }

    pub fn add_in_place(&mut self, other: &Self) {
        assert_same_shape2(&self, &other);

        unsafe {
            for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
                *l = _mm512_add_ps(*l, *r);
            }
        }
    }

    pub fn add_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, &mask);

        unsafe {
            for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
                *l = _mm512_mask_add_ps(*l, *m, *l, *r);
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

        unsafe {
            for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
                *l = _mm512_sub_ps(*l, *r);
            }
        }
    }

    pub fn sub_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        unsafe {
            for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
                *l = _mm512_mask_sub_ps(*l, *m, *l, *r);
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

        unsafe {
            for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
                *l = _mm512_mul_ps(*l, *r);
            }
        }
    }

    pub fn mul_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        unsafe {
            for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
                *l = _mm512_mask_mul_ps(*l, *m, *l, *r);
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

        unsafe {
            for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
                *l = _mm512_div_ps(*l, *r);
            }
        }
    }

    pub fn div_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        unsafe {
            for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
                *l = _mm512_mask_div_ps(*l, *m, *l, *r);
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

        unsafe {
            for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
                *l = _mm512_max_ps(*l, *r);
            }
        }
    }

    pub fn max_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        unsafe {
            for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
                *l = _mm512_mask_max_ps(*l, *m, *l, *r);
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

        unsafe {
            for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
                *l = _mm512_min_ps(*l, *r);
            }
        }
    }

    pub fn min_in_place_masked(&mut self, other: &Self, mask: &Mask<D>) {
        assert_same_shape_with_mask2(&self, &other, mask);

        unsafe {
            for ((l, r), m) in self.data.iter_mut().zip(other.data.iter()).zip(mask.masks.iter()) {
                *l = _mm512_mask_min_ps(*l, *m, *l, *r);
            }
        }
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.add_scalar_in_place(scalar);

        new_array
    }

    pub fn add_scalar_in_place(&mut self, scalar: f32) {
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for d in self.data.iter_mut() {
                *d = _mm512_add_ps(*d, scalar);
            }
        }
    }

    pub fn add_scalar_in_place_masked(&mut self, scalar: f32, mask: &Mask<D>) {
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for (l, m) in self.data.iter_mut().zip(mask.masks.iter()) {
                *l = _mm512_mask_add_ps(*l, *m, *l, scalar);
            }
        }
    }

    pub fn sub_scalar(&self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.sub_scalar_in_place(scalar);

        new_array
    }

    pub fn sub_scalar_in_place(&mut self, scalar: f32) {
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for d in self.data.iter_mut() {
                *d = _mm512_sub_ps(*d, scalar);
            }
        }
    }

    pub fn sub_scalar_in_place_masked(&mut self, scalar: f32, mask: &Mask<D>) {
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for (l, m) in self.data.iter_mut().zip(mask.masks.iter()) {
                *l = _mm512_mask_sub_ps(*l, *m, *l, scalar);
            }
        }
    }

    pub fn mul_scalar(&self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.mul_scalar_in_place(scalar);

        new_array
    }

    pub fn mul_scalar_in_place(&mut self, scalar: f32) {
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for d in self.data.iter_mut() {
                *d = _mm512_mul_ps(*d, scalar);
            }
        }
    }

    pub fn mul_scalar_in_place_masked(&mut self, scalar: f32, mask: &Mask<D>) {
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for (l, m) in self.data.iter_mut().zip(mask.masks.iter()) {
                *l = _mm512_mask_mul_ps(*l, *m, *l, scalar);
            }
        }
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        let mut new_array = self.clone();
        new_array.div_scalar_in_place(scalar);

        new_array
    }

    pub fn div_scalar_in_place(&mut self, scalar: f32) {
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for d in self.data.iter_mut() {
                *d = _mm512_div_ps(*d, scalar);
            }
        }
    }

    pub fn div_scalar_in_place_masked(&mut self, scalar: f32, mask: &Mask<D>) {
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for (l, m) in self.data.iter_mut().zip(mask.masks.iter()) {
                *l = _mm512_mask_div_ps(*l, *m, *l, scalar);
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

        unsafe {
            for ((a, b), c) in a.data.iter().zip(b.data.iter()).zip(self.data.iter_mut()) {
                *c = _mm512_fmadd_ps(*a, *b, *c);
            }
        }
    }

    pub fn fmadd_in_place_masked(&mut self, a: &Self, b: &Self, mask: &Mask<D>)  {
        assert_same_shape_with_mask3(self, a, b, mask);

        unsafe {
            for (((a, b), c), m) in a.data.iter().zip(b.data.iter()).zip(self.data.iter_mut()).zip(mask.masks.iter()) {
                *c = _mm512_mask3_fmadd_ps(*a, *b, *c, *m);
            }
        }
    }

    pub fn sqrt(&self) -> Self {
        let mut new_array = self.clone();
        new_array.sqrt_in_place();

        new_array
    }

    pub fn sqrt_in_place(&mut self) {
        unsafe {
            for d in self.data.iter_mut() {
                *d = _mm512_sqrt_ps(*d);
            }
        }
    }

    pub fn sqrt_in_place_masked(&mut self, mask: &Mask<D>) {
        unsafe {
            for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
                *d = _mm512_mask_sqrt_ps(*d, *m, *d);
            }
        }
    }

    pub fn square(&self) -> Self {
        let mut new_array = self.clone();
        new_array.square_in_place();

        new_array
    }

    pub fn square_in_place(&mut self) {
        unsafe {
            for d in self.data.iter_mut() {
                *d = _mm512_mul_ps(*d, *d);
            }
        }
    }

    pub fn square_in_place_masked(&mut self, mask: &Mask<D>) {
        unsafe {
            for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
                *d = _mm512_mask_mul_ps(*d, *m, *d, *d);
            }
        }
    }

    pub fn abs(&self) -> Self {
        let mut new_array = self.clone();
        new_array.abs_in_place();

        new_array
    }

    pub fn abs_in_place(&mut self) {
        unsafe {
            for d in self.data.iter_mut() {
                *d = _mm512_abs_ps(*d);
            }
        }
    }

    pub fn abs_in_place_masked(&mut self, mask: &Mask<D>) {
        unsafe {
            for (d, m) in self.data.iter_mut().zip(mask.masks.iter()) {
                *d = _mm512_mask_abs_ps(*d, *m, *d);
            }
        }
    }

    // TODO: compare in-place
    fn compare(a: &Array<D>, b: &Array<D>, func: unsafe fn(__m512, __m512) -> __mmask16) -> Mask<D> {
        assert_same_shape2(a, b);
        let mut masks = Vec::with_capacity(a.data.len());

        unsafe {
            for (d1, d2) in a.data.iter().zip(b.data.iter()) {
                masks.push(func(*d1, *d2));
            }
        }

        Mask {
            masks,
            shape: a.shape
        }
    }

    fn compare_in_place(a: &Array<D>, b: &Array<D>, mask: &mut Mask<D>, func: unsafe fn(__m512, __m512) -> __mmask16) {
        assert_same_shape_with_mask2(a, b, mask);

        unsafe {
            for ((d1, d2), m) in a.data.iter().zip(b.data.iter()).zip(mask.masks.iter_mut()) {
                *m = func(*d1, *d2);
            }
        }
    }

    fn compare_scalar(a: &Array<D>, scalar: f32, func: unsafe fn(__m512, __m512) -> __mmask16) -> Mask<D> {
        let scalar = array_to_m512([scalar; 16]);
        let mut masks = Vec::with_capacity(a.data.len());

        unsafe {
            for d in a.data.iter() {
                masks.push(func(*d, scalar));
            }
        }

        Mask {
            masks,
            shape: a.shape
        }
    }

    fn compare_scalar_in_place(a: &Array<D>, scalar: f32, mask: &mut Mask<D>, func: unsafe fn(__m512, __m512) -> __mmask16) {
        assert_eq!(a.shape, mask.shape);
        let scalar = array_to_m512([scalar; 16]);

        unsafe {
            for (d, m) in a.data.iter().zip(mask.masks.iter_mut()) {
                *m = func(*d, scalar);
            }
        }
    }

    pub fn compare_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmpeq_ps_mask)
    }

    pub fn compare_equal_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, _mm512_cmpeq_ps_mask)
    }

    pub fn compare_scalar_equal(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, _mm512_cmpeq_ps_mask)
    }

    pub fn compare_scalar_equal_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, _mm512_cmpeq_ps_mask)
    }

    pub fn compare_not_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmpneq_ps_mask)
    }

    pub fn compare_not_equal_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, _mm512_cmpneq_ps_mask)
    }

    pub fn compare_scalar_not_equal(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, _mm512_cmpneq_ps_mask)
    }

    pub fn compare_scalar_not_equal_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, _mm512_cmpneq_ps_mask)
    }

    pub fn compare_greater_than(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmpnle_ps_mask)
    }

    pub fn compare_greater_than_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, _mm512_cmpnle_ps_mask)
    }

    pub fn compare_scalar_greater_than(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, _mm512_cmpnle_ps_mask)
    }

    pub fn compare_scalar_greater_than_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, _mm512_cmpnle_ps_mask)
    }

    pub fn compare_greater_than_or_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmpnlt_ps_mask)
    }

    pub fn compare_greater_than_or_equal_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, _mm512_cmpnlt_ps_mask)
    }

    pub fn compare_scalar_greater_than_or_equal(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, _mm512_cmpnlt_ps_mask)
    }

    pub fn compare_scalar_greater_than_or_equal_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, _mm512_cmpnlt_ps_mask)
    }

    pub fn compare_less_than(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmplt_ps_mask)
    }

    pub fn compare_less_than_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, _mm512_cmplt_ps_mask)
    }

    pub fn compare_scalar_less_than(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, _mm512_cmplt_ps_mask)
    }

    pub fn compare_scalar_less_than_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, _mm512_cmplt_ps_mask)
    }

    pub fn compare_less_than_or_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmple_ps_mask)
    }

    pub fn compare_less_than_or_equal_in_place(&self, other: &Self, mask: &mut Mask<D>) {
        Self::compare_in_place(self, other, mask, _mm512_cmple_ps_mask)
    }

    pub fn compare_scalar_less_than_or_equal(&self, scalar: f32) -> Mask<D> {
        Self::compare_scalar(self, scalar, _mm512_cmple_ps_mask)
    }

    pub fn compare_scalar_less_than_or_equal_in_place(&self, scalar: f32, mask: &mut Mask<D>) {
        Self::compare_scalar_in_place(self, scalar, mask, _mm512_cmple_ps_mask)
    }
    
    pub fn exp(&self) -> Self {
        let mut tmp = self.clone();
        tmp.exp_in_place();

        tmp
    }

    pub fn exp_in_place(&mut self) {
        // adapted from https://stackoverflow.com/a/49090523

        let l2e = array_to_m512([1.442695041f32; 16]); // log2(e)
        let l2h = array_to_m512([-6.93145752e-1f32; 16]); // -log(2)_hi
        let l2l = array_to_m512([-1.42860677e-6f32; 16]); // -log(2)_lo
        // coefficients for core approximation to exp() in [-log(2)/2, log(2)/2]
        let c0 = array_to_m512([0.041944388f32; 16]);
        let c1 = array_to_m512([0.168006673f32; 16]);
        let c2 = array_to_m512([0.499999940f32; 16]);
        let c3 = array_to_m512([0.999956906f32; 16]);
        let c4 = array_to_m512([0.999999642f32; 16]);

        unsafe {
            for x in self.data.iter_mut() {
                // exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i
                let t = _mm512_mul_ps(*x, l2e);
                let mut r = _mm512_mul_round_ps(*x, l2e, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); // r = rint (t)
                let mut f = _mm512_fmadd_ps(r, l2h, *x); // x - log(2)_hi * r
                f = _mm512_fmadd_ps(r, l2l, f); // f = x - log(2)_hi * r - log(2)_lo * r

                let i = _mm512_cvtps_epi32(t); // i = (int)rint(t)

                // p ~= exp (f), -log(2)/2 <= f <= log(2)/2
                let mut p = c0; // c0
                p = _mm512_fmadd_ps(p, f, c1); // c0*f+c1
                p = _mm512_fmadd_ps(p, f, c2); // (c0*f+c1)*f+c2
                p = _mm512_fmadd_ps(p, f, c3); // ((c0*f+c1)*f+c2)*f+c3
                p = _mm512_fmadd_ps(p, f, c4); // (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f)

                // exp(x) = 2^i * p
                let j = _mm512_slli_epi32(i, 23); // i << 23
                r = _mm512_castsi512_ps(_mm512_add_epi32(j, _mm512_castps_si512(p))); // r = p * 2^i

                *x = r;
            }
        }
    }

    // TODO: exp_in_place_masked
}

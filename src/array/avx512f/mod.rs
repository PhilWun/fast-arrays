mod one_dimension;
mod two_dimensions;

use std::{
    arch::x86_64::{__m512, _mm512_add_ps, _mm512_sub_ps, _mm512_mul_ps, _mm512_div_ps, _mm512_max_ps, _mm512_min_ps, _mm512_sqrt_ps, _mm512_fmadd_ps, _mm512_abs_ps, _mm512_cmpeq_ps_mask, _mm512_cmpneq_ps_mask, _mm512_cmpnle_ps_mask, _mm512_cmpnlt_ps_mask, _mm512_cmplt_ps_mask, _mm512_cmple_ps_mask, _mm512_mul_round_ps, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_NO_EXC, _mm512_cvtps_epi32, _mm512_slli_epi32, _mm512_castsi512_ps, _mm512_add_epi32, _mm512_castps_si512, __mmask16},
    simd::f32x16
};

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

fn assert_same_shape3<const D: usize>(a: &Array<D>, b: &Array<D>, c: &Array<D>) {
    assert_eq!(a.shape, b.shape, "the lengths of array one and two don't match: {:?} != {:?}", a.shape, b.shape);
    assert_eq!(b.shape, c.shape, "the lengths of array two and three don't match: {:?} != {:?}", b.shape, c.shape);
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

impl<const D: usize> Array<D> {
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

    pub fn compare_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmpeq_ps_mask)
    }

    pub fn compare_not_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmpneq_ps_mask)
    }

    pub fn compare_greater_than(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmpnle_ps_mask)
    }

    pub fn compare_greater_than_or_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmpnlt_ps_mask)
    }

    pub fn compare_less_than(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmplt_ps_mask)
    }

    pub fn compare_less_than_or_equal(&self, other: &Self) -> Mask<D> {
        Self::compare(self, other, _mm512_cmple_ps_mask)
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
}

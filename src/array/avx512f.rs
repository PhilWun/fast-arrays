use std::{
    arch::x86_64::{__m512, _mm512_add_ps, _mm512_sub_ps, _mm512_mul_ps, _mm512_div_ps, _mm512_max_ps, _mm512_min_ps, _mm512_sqrt_ps, _mm512_fmadd_ps, _mm512_abs_ps, _mm512_cmpeq_ps_mask, _mm512_cmpneq_ps_mask, _mm512_cmpnle_ps_mask, _mm512_cmpnlt_ps_mask, _mm512_cmplt_ps_mask, _mm512_cmple_ps_mask, _mm512_mul_round_ps, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_NO_EXC, _mm512_cvtps_epi32, _mm512_slli_epi32, _mm512_castsi512_ps, _mm512_add_epi32, _mm512_castps_si512, __mmask16, _mm512_mask_add_ps, _mm512_reduce_add_ps, _mm512_mask_mul_ps, _mm512_reduce_mul_ps},
    simd::f32x16
};

use crate::{Array1D, Mask1D};

fn m512_to_array(value: __m512) -> [f32; 16] {
    let value: f32x16 = value.into();
    value.into()
}

fn array_to_m512(value: [f32; 16]) -> __m512 {
    let value: f32x16 = value.into();
    value.into()
}

fn assert_same_lengths2(a: &Array1D, b: &Array1D) {
    assert_eq!(a.len, b.len, "the lengths of array one and two don't match: {} != {}", a.len, b.len);
}

fn assert_same_lengths3(a: &Array1D, b: &Array1D, c: &Array1D) {
    assert_eq!(a.len, b.len, "the lengths of array one and two don't match: {} != {}", a.len, b.len);
    assert_eq!(b.len, c.len, "the lengths of array two and three don't match: {} != {}", b.len, c.len);
}

impl From<Array1D> for Vec<f32> {
    fn from(value: Array1D) -> Self {
        let mut converted = vec![0f32; value.len];
        let mut index: usize = 0;

        for register in value.data {
            let register = m512_to_array(register);

            for i in 0..16 {
                if index >= value.len {
                    break;
                }

                converted[index] = register[i];
                index += 1;
            }
        }

        converted
    }
}

impl From<Vec<f32>> for Array1D {
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

        Array1D {
            data: data,
            len: value.len(),
        }
    }
}

impl Array1D {
    pub fn zeros(len: usize) -> Self {
        let register_count = len.div_ceil(16);
        let zero = array_to_m512([0f32; 16]);
        let data = vec![zero; register_count];

        Self {
            data,
            len,
        }
    }

    pub fn get(&self, index: usize) -> f32 {
        if index >= self.len {
            panic!("tried to get index {}, but the array has only {} element(s)", index, self.len);
        }

        let register_index = index / 16;
        let value_index = index % 16;

        let value = m512_to_array(self.data[register_index])[value_index];

        value
    }

    pub fn set(&mut self, index: usize, value: f32) {
        if index >= self.len {
            panic!("tried to set index {}, but the array has only {} element(s)", index, self.len);
        }

        let register_index = index / 16;
        let value_index = index % 16;

        let mut new_register = m512_to_array(self.data[register_index]);
        new_register[value_index] = value;

        self.data[register_index] = array_to_m512(new_register);
    }

    pub fn add(&self, other: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.add_in_place(other);

        new_array
    }

    pub fn add_in_place(&mut self, other: &Self) {
        assert_same_lengths2(&self, &other);

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
        assert_same_lengths2(&self, &other);

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
        assert_same_lengths2(&self, &other);

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
        assert_same_lengths2(&self, &other);

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
        assert_same_lengths2(self, other);

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
        assert_same_lengths2(self, other);

        unsafe {
            for (l, r) in self.data.iter_mut().zip(other.data.iter()) {
                *l = _mm512_min_ps(*l, *r);
            }
        }
    }

    pub fn fmadd(&self, a: &Self, b: &Self) -> Self {
        let mut new_array = self.clone();
        new_array.fmadd_in_place(a, b);

        new_array
    }

    pub fn fmadd_in_place(&mut self, a: &Self, b: &Self)  {
        assert_same_lengths3(self, a, b);

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

    fn compare(a: &Array1D, b: &Array1D, func: unsafe fn(__m512, __m512) -> __mmask16) -> Mask1D {
        assert_same_lengths2(a, b);
        let mut masks = Vec::with_capacity(a.data.len());

        unsafe {
            for (d1, d2) in a.data.iter().zip(b.data.iter()) {
                masks.push(func(*d1, *d2));
            }
        }

        Mask1D {
            masks,
            len: a.len
        }
    }

    pub fn compare_equal(&self, other: &Self) -> Mask1D {
        Self::compare(self, other, _mm512_cmpeq_ps_mask)
    }

    pub fn compare_not_equal(&self, other: &Self) -> Mask1D {
        Self::compare(self, other, _mm512_cmpneq_ps_mask)
    }

    pub fn compare_greater_than(&self, other: &Self) -> Mask1D {
        Self::compare(self, other, _mm512_cmpnle_ps_mask)
    }

    pub fn compare_greater_than_or_equal(&self, other: &Self) -> Mask1D {
        Self::compare(self, other, _mm512_cmpnlt_ps_mask)
    }

    pub fn compare_less_than(&self, other: &Self) -> Mask1D {
        Self::compare(self, other, _mm512_cmplt_ps_mask)
    }

    pub fn compare_less_than_or_equal(&self, other: &Self) -> Mask1D {
        Self::compare(self, other, _mm512_cmple_ps_mask)
    }
    
    pub fn exp(&self) -> Array1D {
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

    pub fn sum(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }

        let mut sum_register = array_to_m512([0.0; 16]);
        let mut last_register_mask = 0xFFFF;

        if self.len % 16 != 0 {
            last_register_mask = 0xFFFF >> (16 - (self.len % 16));
        }

        unsafe {
            for d in self.data[0..self.data.len() - 1].iter() {
                sum_register = _mm512_add_ps(sum_register, *d);
            }

            sum_register = _mm512_mask_add_ps(sum_register, last_register_mask, sum_register, *self.data.last().unwrap());

            _mm512_reduce_add_ps(sum_register)
        }
    }

    pub fn product(&self) -> f32 {
        if self.len == 0 {
            return 1.0;
        }

        let mut sum_register = array_to_m512([1.0; 16]);
        let mut last_register_mask = 0xFFFF;

        if self.len % 16 != 0 {
            last_register_mask = 0xFFFF >> (16 - (self.len % 16));
        }

        unsafe {
            for d in self.data[0..self.data.len() - 1].iter() {
                sum_register = _mm512_mul_ps(sum_register, *d);
            }

            sum_register = _mm512_mask_mul_ps(sum_register, last_register_mask, sum_register, *self.data.last().unwrap());

            _mm512_reduce_mul_ps(sum_register)
        }
    }

    pub fn dot_product(&self, other: &Self) -> f32 {
        assert_same_lengths2(self, other);

        if self.len == 0 {
            return 0.0;
        }

        let mut sum_register = array_to_m512([0.0; 16]);
        let mut last_register_mask = 0xFFFF;

        if self.len % 16 != 0 {
            last_register_mask = 0xFFFF >> (16 - (self.len % 16));
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

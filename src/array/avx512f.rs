use std::{
    arch::x86_64::{__m512, _mm512_add_ps, _mm512_sub_ps, _mm512_mul_ps, _mm512_div_ps, _mm512_max_ps, _mm512_min_ps, _mm512_sqrt_ps, _mm512_fmadd_ps},
    ops::{Add, Sub, Mul, Div},
    simd::f32x16
};

use super::Array;

fn m512_to_array(value: __m512) -> [f32; 16] {
    let value: f32x16 = value.into();
    value.into()
}

fn array_to_m512(value: [f32; 16]) -> __m512 {
    let value: f32x16 = value.into();
    value.into()
}

impl Array<1> {
    pub fn zeros(shape: usize) -> Self {
        let register_count = shape.div_ceil(16);
        let zero = array_to_m512([0f32; 16]);
        let data = vec![zero; register_count];

        Self {
            data,
            shape: [shape],
        }
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        if index >= self.shape[0] {
            return None;
        }

        let register_index = index / 16;
        let value_index = index % 16;

        let value = m512_to_array(self.data[register_index])[value_index];

        Some(value)
    }

    pub fn set(&mut self, index: usize, value: f32) -> Option<()> {
        if index >= self.shape[0] {
            return None;
        }

        let register_index = index / 16;
        let value_index = index % 16;

        let mut new_register = m512_to_array(self.data[register_index]);
        new_register[value_index] = value;

        self.data[register_index] = array_to_m512(new_register);

        Some(())
    }

    pub fn max(&self, other: &Self) -> Result<Self, ()> {
        if self.shape[0] != other.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        unsafe {
            for (l, r) in self.data.iter().zip(other.data.iter()) {
                new_data.push(_mm512_max_ps(*l, *r));
            }
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone()
        })
    }

    pub fn min(&self, other: &Self) -> Result<Self, ()> {
        if self.shape[0] != other.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        unsafe {
            for (l, r) in self.data.iter().zip(other.data.iter()) {
                new_data.push(_mm512_min_ps(*l, *r));
            }
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone()
        })
    }

    pub fn sqrt(&self) -> Self {
        let mut new_data = Vec::with_capacity(self.data.len());

        unsafe {
            for d in self.data.iter() {
                new_data.push(_mm512_sqrt_ps(*d));
            }
        }

        Self {
            data: new_data,
            shape: self.shape.clone()
        }
    }

    pub fn fmadd(&self, a: &Self, b: &Self) -> Result<Self, ()> {
        if self.shape[0] != a.shape[0] {
            return Err(());
        }

        if self.shape[0] != b.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        unsafe {
            for ((a, b), c) in a.data.iter().zip(b.data.iter()).zip(self.data.iter()) {
                new_data.push(_mm512_fmadd_ps(*a, *b, *c));
            }
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone()
        })
    }

    pub fn fmadd_in_place(&mut self, a: &Self, b: &Self) -> Result<(), ()> {
        if self.shape[0] != a.shape[0] {
            return Err(());
        }

        if self.shape[0] != b.shape[0] {
            return Err(());
        }

        unsafe {
            for ((a, b), c) in a.data.iter().zip(b.data.iter()).zip(self.data.iter_mut()) {
                *c = _mm512_fmadd_ps(*a, *b, *c);
            }
        }

        Ok(())
    }
}

impl Add for Array<1> {
    type Output = Result<Self, ()>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape[0] != rhs.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        unsafe {
            for (l, r) in self.data.iter().zip(rhs.data.iter()) {
                new_data.push(_mm512_add_ps(*l, *r));
            }
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }
}

impl Sub for Array<1> {
    type Output = Result<Self, ()>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape[0] != rhs.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        unsafe {
            for (l, r) in self.data.iter().zip(rhs.data.iter()) {
                new_data.push(_mm512_sub_ps(*l, *r));
            }
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }
}

impl Mul for Array<1> {
    type Output = Result<Self, ()>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape[0] != rhs.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        unsafe {
            for (l, r) in self.data.iter().zip(rhs.data.iter()) {
                new_data.push(_mm512_mul_ps(*l, *r));
            }
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }
}

impl Div for Array<1> {
    type Output = Result<Self, ()>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.shape[0] != rhs.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        unsafe {
            for (l, r) in self.data.iter().zip(rhs.data.iter()) {
                new_data.push(_mm512_div_ps(*l, *r));
            }
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_1d_small() {
        let array = Array::<1>::zeros(3);

        assert_eq!(array.shape, [3]);
        assert_eq!(array.data.len(), 1);

        let data = m512_to_array(array.data[0]);

        assert_eq!(data[0..3], vec![0f32; 3]);
    }

    #[test]
    fn zeros_1d_one_full_register() {
        let array = Array::<1>::zeros(16);

        assert_eq!(array.shape, [16]);
        assert_eq!(array.data.len(), 1);

        let data = m512_to_array(array.data[0]);

        assert_eq!(data, [0f32; 16]);
    }

    #[test]
    fn zeros_1d_two_registers() {
        let array = Array::<1>::zeros(17);

        assert_eq!(array.shape, [17]);
        assert_eq!(array.data.len(), 2);

        let data1 = m512_to_array(array.data[0]);
        let data2 = m512_to_array(array.data[1]);

        assert_eq!(data1, [0f32; 16]);
        assert_eq!(data2[0..1], [0f32; 1]);
    }
}

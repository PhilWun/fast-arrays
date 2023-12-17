#![feature(portable_simd)]
#![feature(stdsimd)]
use std::{simd::{Simd, f32x16}, arch::x86_64::{_mm512_fmadd_ps, __m512}, time::Instant};

pub fn example() {
    let c = [3f32; 16];

    let mut a = __m512::from(Simd::from(c));
    let mut b = a.clone();
    let mut c = a.clone();
    let mut d = a.clone();

    let time1 = Instant::now();

    unsafe {
        for _ in 0..1_000_000 {
            a = _mm512_fmadd_ps(a, a, a);
            b = _mm512_fmadd_ps(b, b, b);
            c = _mm512_fmadd_ps(c, c, c);
            d = _mm512_fmadd_ps(d, d, d);
        }
    }

    let time2 = Instant::now();

    for result in Simd::from(a).as_array() {
        print!("{}, ", result);
    }

    for result in Simd::from(b).as_array() {
        print!("{}, ", result);
    }

    for result in Simd::from(c).as_array() {
        print!("{}, ", result);
    }

    for result in Simd::from(d).as_array() {
        print!("{}, ", result);
    }

    println!();
    println!("{}s", (time2 - time1).as_secs_f32());
}

fn m512_to_array(value: __m512) -> [f32; 16] {
    let value: f32x16 = value.into();
    value.into()
}

fn array_to_m512(value: [f32; 16]) -> __m512 {
    let value: f32x16 = value.into();
    value.into()
}

struct Array<const D: usize> {
    data: Vec<__m512>,
    shape: [usize; D]
}

impl Array<1> {
    pub fn zeros(shape: usize) -> Self {
        let register_count = shape.div_ceil(16);
        let zero = array_to_m512([0f32; 16]);
        let data = vec![zero; register_count];

        Self {
            data,
            shape: [shape]
        }
    }
}

impl From<Array<1>> for Vec<f32> {
    fn from(value: Array<1>) -> Self {
        let mut converted = vec![0f32; value.shape[0]];
        let mut index: usize = 0;

        for register in value.data {
            let register = m512_to_array(register);

            while index < converted.len() {
                converted[index] = register[index % 16];
                index += 1;
            }
        }

        converted
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
        let data2 = m512_to_array(array.data[0]);

        assert_eq!(data1, [0f32; 16]);
        assert_eq!(data2[0..1], [0f32; 1]);
    }
}

#![feature(portable_simd)]
#![feature(stdsimd)]
use std::{
    arch::x86_64::{__m512, _mm512_fmadd_ps},
    simd::{f32x16, Simd},
    time::Instant,
};

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

pub struct Array<const D: usize> {
    data: Vec<__m512>,
    shape: [usize; D],
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
        let data2 = m512_to_array(array.data[0]);

        assert_eq!(data1, [0f32; 16]);
        assert_eq!(data2[0..1], [0f32; 1]);
    }

    #[test]
    fn conversion_small() {
        let value = vec![0.0, 1.0, 2.0];
        let converted: Array<1> = value.clone().into();
        let converted_back: Vec<f32> = converted.into();

        assert_eq!(converted_back, value);
    }

    #[test]
    fn conversion_one_full_register() {
        let value = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let converted: Array<1> = value.clone().into();
        let converted_back: Vec<f32> = converted.into();

        assert_eq!(converted_back, value);
    }

    #[test]
    fn conversion_two_register() {
        let value = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let converted: Array<1> = value.clone().into();
        let converted_back: Vec<f32> = converted.into();

        assert_eq!(converted_back, value);
    }

    #[test]
    fn get() {
        let array: Array<1> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ]
        .into();

        assert_eq!(array.get(0), Some(0.0));
        assert_eq!(array.get(15), Some(15.0));
        assert_eq!(array.get(16), Some(16.0));
        assert_eq!(array.get(17), None);
    }

    #[test]
    fn set() {
        let mut array: Array<1> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ]
        .into();

        assert_eq!(array.set(0, 42.0), Some(()));
        assert_eq!(array.set(15, 37.5), Some(()));
        assert_eq!(array.set(16, 31.9), Some(()));
        assert_eq!(array.set(17, 95.4), None);

        let data: Vec<f32> = array.into();
        assert_eq!(
            data,
            vec![
                42.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                37.5, 31.9
            ]
        );
    }
}

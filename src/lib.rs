#![feature(portable_simd)]
#![feature(stdsimd)]

mod array;
mod mask;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::__m512;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::__mmask16;

#[derive(Clone)]
pub struct Array1D {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    data: Vec<__m512>,
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    data: Vec<f32>,
    len: usize,
}

#[derive(Clone)]
pub struct Mask1D {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    masks: Vec<__mmask16>,
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    data: Vec<bool>,
    len: usize,
}

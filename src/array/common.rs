#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::__m512;

#[derive(Clone)]
pub struct Array1D {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    pub(super) data: Vec<__m512>,
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    pub(super) data: Vec<f32>,
    pub(super) len: usize,
}

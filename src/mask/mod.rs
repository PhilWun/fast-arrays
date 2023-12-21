#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512f;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod fallback;

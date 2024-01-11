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

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::__mmask16;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx512f;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
mod fallback;

#[derive(Clone)]
pub struct Mask<const D: usize> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    masks: Vec<__mmask16>,
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    masks: Vec<bool>,
    shape: [usize; D],
}

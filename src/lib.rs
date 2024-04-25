#![feature(portable_simd)]
#![feature(stdsimd)]

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

mod array;
mod mask;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::__m512;
use std::ops::Deref;

pub use mask::Mask;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub struct Array<const D: usize, C>
{
    data: C,
    shape: [usize; D],
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[derive(Clone)]
pub struct Array<const D: usize, C>
where
    C: Deref<Target = [f32]>,
{
    data: Vec<f32>,
    shape: [usize; D],
}

impl<const D: usize, C> Array<D, C>
where
    C: Deref<Target = [__m512]>
{
    fn clone(&self) -> Array<D, Vec<__m512>> {
        Array { data: Vec::from_iter(self.data.iter().map(|x| *x)), shape: self.shape.clone() }
    }
}

impl<const D: usize, C> Array<D, C>
where
    C: Deref<Target = [__m512]>,
{
    pub fn number_of_elements(&self) -> usize {
        let mut elements_count = 1;

        for s in self.shape.iter() {
            elements_count *= s;
        }

        elements_count
    }
}

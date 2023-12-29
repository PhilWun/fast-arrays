/**
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

use crate::Mask;

impl From<Mask<1>> for Vec<bool> {
    fn from(value: Mask<1>) -> Self {
        value.data
    }
}

impl From<Vec<bool>> for Mask<1> {
    fn from(value: Vec<bool>) -> Self {
        let len = value.len();
        Self {
            data: value,
            shape: [len],
        }
    }
}

impl Mask<1> {
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![false; len],
            shape: [len]
        }
    }
}
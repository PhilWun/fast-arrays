use std::arch::x86_64::__mmask16;

use crate::Mask1D;

impl From<Mask1D> for Vec<bool> {
    fn from(value: Mask1D) -> Self {
        let mut converted = vec![false; value.len];
        let mut index: usize = 0;

        for register in value.masks {
            for i in 0..16 {
                if index >= value.len {
                    break;
                }

                converted[index] = register & (1 << i) > 0;
                index += 1;
            }
        }

        converted
    }
}

impl From<Vec<bool>> for Mask1D {
    fn from(value: Vec<bool>) -> Self {
        let register_count = value.len().div_ceil(16);
        let mut masks: Vec<__mmask16> = Vec::with_capacity(register_count);
        let mut index = 0;

        for _ in 0..register_count {
            let mut new_register_data: __mmask16 = 0;

            for i in 0..16 {
                if index < value.len() {
                    new_register_data |= (value[index] as __mmask16) << i;
                    index += 1;
                }
            }

            masks.push(new_register_data);
        }

        Self {
            masks,
            len: value.len(),
        }
    }
}

impl Mask1D {
    pub fn new(len: usize) -> Self {
        let mask_count = len.div_ceil(16);
        let masks = vec![0u16; len];

        Self {
            masks,
            len: mask_count
        }
    }
}

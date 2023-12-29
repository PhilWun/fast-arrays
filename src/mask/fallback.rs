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

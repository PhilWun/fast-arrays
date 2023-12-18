use std::ops::Add;

use super::Array;

impl Array<1> {
    pub fn zeros(shape: usize) -> Self {
        Self {
            data: vec![0.0; shape],
            shape: [shape]
        }
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).map(|x| *x)
    }

    pub fn set(&mut self, index: usize, value: f32) -> Option<()> {
        self.data.get_mut(index).map(|x| {*x = value})
    }
}

impl Add for Array<1> {
    type Output = Result<Self, ()>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape[0] != rhs.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            new_data.push(*l + *r);
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }
}

impl From<Array<1>> for Vec<f32> {
    fn from(value: Array<1>) -> Self {
        value.data.clone()
    }
}

impl From<Vec<f32>> for Array<1> {
    fn from(value: Vec<f32>) -> Self {
        let len = value.len();

        Array {
            data: value,
            shape: [len],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros() {
        let array = Array::<1>::zeros(42);

        assert_eq!(array.shape, [42]);
        assert_eq!(array.data.len(), 42);
        assert_eq!(array.data, vec![0.0; 42]);
    }
}

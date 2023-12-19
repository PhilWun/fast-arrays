use std::ops::{Add, Sub, Mul, Div};

use super::Array;

impl Array<1> {
    pub fn zeros(shape: usize) -> Self {
        Self {
            data: vec![0.0; shape],
            shape: [shape],
        }
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).map(|x| *x)
    }

    pub fn set(&mut self, index: usize, value: f32) -> Option<()> {
        self.data.get_mut(index).map(|x| *x = value)
    }

    pub fn max(&self, other: &Self) -> Result<Self, ()> {
        if self.shape[0] != other.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(other.data.iter()) {
            new_data.push(l.max(*r));
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }

    pub fn min(&self, other: &Self) -> Result<Self, ()> {
        if self.shape[0] != other.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(other.data.iter()) {
            new_data.push(l.min(*r));
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }

    pub fn sqrt(&self) -> Self {
        let mut new_data = Vec::with_capacity(self.data.len());

        for v in self.data.iter() {
            new_data.push(v.sqrt());
        }

        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn fmadd(&self, a: &Self, b: &Self) -> Result<Self, ()> {
        if self.shape[0] != a.shape[0] {
            return Err(());
        }

        if self.shape[0] != b.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for ((a, b), c) in a.data.iter().zip(b.data.iter()).zip(self.data.iter()) {
            new_data.push(a * b + c);
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone()
        })
    }

    pub fn fmadd_in_place(&mut self, a: &Self, b: &Self) -> Result<(), ()> {
        if self.shape[0] != a.shape[0] {
            return Err(());
        }

        if self.shape[0] != b.shape[0] {
            return Err(());
        }

        for ((a, b), c) in a.data.iter().zip(b.data.iter()).zip(self.data.iter_mut()) {
            *c += a * b;
        }

        Ok(())
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

impl Sub for Array<1> {
    type Output = Result<Self, ()>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape[0] != rhs.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            new_data.push(*l - *r);
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }
}

impl Mul for Array<1> {
    type Output = Result<Self, ()>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape[0] != rhs.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            new_data.push(*l * *r);
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }
}

impl Div for Array<1> {
    type Output = Result<Self, ()>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.shape[0] != rhs.shape[0] {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            new_data.push(*l / *r);
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

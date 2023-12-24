use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign};

use crate::Array1D;

impl From<Array1D> for Vec<f32> {
    fn from(value: Array1D) -> Self {
        value.data.clone()
    }
}

impl From<Vec<f32>> for Array1D {
    fn from(value: Vec<f32>) -> Self {
        let len = value.len();

        Array1D {
            data: value,
            len: len,
        }
    }
}

impl Array1D {
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
            len: len,
        }
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).map(|x| *x)
    }

    pub fn set(&mut self, index: usize, value: f32) -> Option<()> {
        self.data.get_mut(index).map(|x| *x = value)
    }

    pub fn max(&self, other: &Self) -> Result<Self, ()> {
        if self.len != other.len {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(other.data.iter()) {
            new_data.push(l.max(*r));
        }

        Ok(Self {
            data: new_data,
            len: self.len,
        })
    }

    pub fn min(&self, other: &Self) -> Result<Self, ()> {
        if self.len != other.len {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(other.data.iter()) {
            new_data.push(l.min(*r));
        }

        Ok(Self {
            data: new_data,
            len: self.len,
        })
    }

    pub fn sqrt(&self) -> Self {
        let mut new_data = Vec::with_capacity(self.data.len());

        for v in self.data.iter() {
            new_data.push(v.sqrt());
        }

        Self {
            data: new_data,
            len: self.len,
        }
    }

    pub fn fmadd(&self, a: &Self, b: &Self) -> Result<Self, ()> {
        if self.len != a.len {
            return Err(());
        }

        if self.len != b.len {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for ((a, b), c) in a.data.iter().zip(b.data.iter()).zip(self.data.iter()) {
            new_data.push(a * b + c);
        }

        Ok(Self {
            data: new_data,
            len: self.len
        })
    }

    pub fn fmadd_in_place(&mut self, a: &Self, b: &Self) -> Result<(), ()> {
        if self.len != a.len {
            return Err(());
        }

        if self.len != b.len {
            return Err(());
        }

        for ((a, b), c) in a.data.iter().zip(b.data.iter()).zip(self.data.iter_mut()) {
            *c += a * b;
        }

        Ok(())
    }
}

impl Add for Array1D {
    type Output = Result<Self, ()>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.len != rhs.len {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            new_data.push(*l + *r);
        }

        Ok(Self {
            data: new_data,
            len: self.len,
        })
    }
}

impl AddAssign for Array1D {
    fn add_assign(&mut self, rhs: Self) {
        if self.len != rhs.len {
            panic!();
        }

        for (l, r) in self.data.iter_mut().zip(rhs.data.iter()) {
            *l += r;
        }
    }
}

impl Sub for Array1D {
    type Output = Result<Self, ()>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.len != rhs.len {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            new_data.push(*l - *r);
        }

        Ok(Self {
            data: new_data,
            len: self.len,
        })
    }
}

impl SubAssign for Array1D {
    fn sub_assign(&mut self, rhs: Self) {
        if self.len != rhs.len {
            panic!();
        }

        for (l, r) in self.data.iter_mut().zip(rhs.data.iter()) {
            *l -= r;
        }
    }
}

impl Mul for Array1D {
    type Output = Result<Self, ()>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.len != rhs.len {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            new_data.push(*l * *r);
        }

        Ok(Self {
            data: new_data,
            len: self.len,
        })
    }
}

impl MulAssign for Array1D {
    fn mul_assign(&mut self, rhs: Self) {
        if self.len != rhs.len {
            panic!();
        }

        for (l, r) in self.data.iter_mut().zip(rhs.data.iter()) {
            *l *= r;
        }
    }
}

impl Div for Array1D {
    type Output = Result<Self, ()>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.len != rhs.len {
            return Err(());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for (l, r) in self.data.iter().zip(rhs.data.iter()) {
            new_data.push(*l / *r);
        }

        Ok(Self {
            data: new_data,
            len: self.len,
        })
    }
}

impl DivAssign for Array1D {
    fn div_assign(&mut self, rhs: Self) {
        if self.len != rhs.len {
            panic!();
        }

        for (l, r) in self.data.iter_mut().zip(rhs.data.iter()) {
            *l /= r;
        }
    }
}

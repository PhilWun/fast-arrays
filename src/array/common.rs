#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::__m512;

#[derive(Clone)]
pub struct Array<const D: usize> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    pub(super) data: Vec<__m512>,
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    pub(super) data: Vec<f32>,
    pub(super) shape: [usize; D],
}

type Type1Function = fn(Array<1>, Array<1>) -> Array<1>;
type Type2Function = fn(&mut Array<1>, Array<1>);
type Type3Function = fn(&Array<1>, &Array<1>) -> Array<1>;
type Type4Function = fn(&Array<1>) -> Array<1>;
type Type5Function = fn(&Array<1>, &Array<1>, &Array<1>) -> Array<1>;
type Type6Function = fn(&mut Array<1>, &Array<1>, &Array<1>);

#[cfg(test)]
mod tests {
    use std::ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign};

    use rstest::rstest;

    use super::*;

    #[test]
    fn conversion() {
        let mut values = Vec::new();

        for i in 0..64 {
            let converted: Array<1> = values.clone().into();
            let converted_back: Vec<f32> = converted.into();

            assert_eq!(converted_back, values);

            values.push(i as f32);
        }
    }

    #[test]
    fn zeros() {
        let mut values = Vec::new();

        for i in 0..64 {
            let zeros: Vec<f32> = Array::<1>::zeros(i).into();

            assert_eq!(zeros, values);

            values.push(0.0f32);
        }
    }

    #[test]
    fn get() {
        let mut data = Vec::new();

        for i in 0..64 {
            data.push(i as f32);
        }

        let array: Array<1> = data.clone().into();

        for i in 0..64 {
            assert_eq!(array.get(i), *data.get(i).unwrap());
        }
    }

    #[test]
    #[should_panic]
    fn get_out_of_bounds() {
        let mut data = Vec::new();

        for i in 0..64 {
            data.push(i as f32);
        }

        let array: Array<1> = data.clone().into();
        array.get(64);
    }

    #[test]
    fn set() {
        let mut data = Vec::new();

        for i in 0..64 {
            data.push(i as f32);
        }

        let mut array: Array<1> = data.clone().into();

        for i in 0..64 {
            array.set(i, (i + 10) as f32);
            assert_eq!(array.get(i), (i + 10) as f32);
        }
    }

    #[test]
    #[should_panic]
    fn set_out_of_bounds() {
        let mut data = Vec::new();

        for i in 0..64 {
            data.push(i as f32);
        }

        let mut array: Array<1> = data.clone().into();
        array.set(64, 42.0);
    }

    #[rstest]
    #[case::add(Add::add, Add::add)]
    #[case::sub(Sub::sub, Sub::sub)]
    #[case::mul(Mul::mul, Mul::mul)]
    #[case::div(Div::div, Div::div)]
    fn type1(#[case] test_function: Type1Function, #[case] target_function: fn(f32, f32) -> f32) {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            let result: Vec<f32> = test_function(array1, array2).into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, target_function(*d1, *d2));
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[rstest]
    #[case::add(Add::add)]
    #[case::sub(Sub::sub)]
    #[case::mul(Mul::mul)]
    #[case::div(Div::div)]
    #[should_panic]
    fn type1_shape_mismatch(#[case] test_function: Type1Function) {
        let array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let _ = test_function(array1, array2);
    }

    #[rstest]
    #[case::add(AddAssign::add_assign, Add::add)]
    #[case::sub(SubAssign::sub_assign, Sub::sub)]
    #[case::mul(MulAssign::mul_assign, Mul::mul)]
    #[case::div(DivAssign::div_assign, Div::div)]
    fn type2(#[case] test_function: Type2Function, #[case] target_function: fn(f32, f32) -> f32) {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let mut array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            test_function(&mut array1, array2);
            let result: Vec<f32> = array1.into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, target_function(*d1, *d2));
            }

            data1.push(i as f32); // TODO: use pseudo random numbers with a fixed seed
            data2.push((i + 1) as f32);
        }
    }

    #[rstest]
    #[case::sub(AddAssign::add_assign)]
    #[case::add(SubAssign::sub_assign)]
    #[case::mul(MulAssign::mul_assign)]
    #[case::div(DivAssign::div_assign)]
    #[should_panic]
    fn type2_shape_mismatch(#[case] test_function: Type2Function) {
        let mut array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let _ = test_function(&mut array1, array2);
    }

    #[rstest]
    #[case::max(Array::<1>::max, f32::max)]
    #[case::min(Array::<1>::min, f32::min)]
    fn type3(#[case] test_function: Type3Function, #[case] target_function: fn(f32, f32) -> f32) {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            let result: Vec<f32> = test_function(&array1, &array2).into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, target_function(*d1, *d2));
            }

            data1.push(i as f32); // TODO: use pseudo random numbers with a fixed seed
            data2.push((i + 1) as f32);
        }
    }

    #[rstest]
    #[case::max(Array::<1>::max)]
    #[case::min(Array::<1>::min)]
    #[should_panic]
    fn type3_shape_mismatch(#[case] test_function: Type3Function) {
        let array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let _ = test_function(&array1, &array2);
    }

    #[rstest]
    #[case::sqrt(Array::<1>::sqrt)]
    fn type4(#[case] test_function: Type4Function) {
        let mut data = Vec::new();

        for i in 0..64 {
            let array: Array<1> = data.clone().into();
            let result: Vec<f32> = test_function(&array).into();

            for (d, r) in data.iter().zip(result.iter()) {
                assert_eq!(*r, d.sqrt());
            }

            data.push(i as f32);
        }
    }

    #[rstest]
    #[case::fmadd(Array::<1>::fmadd, |x, y, z| y * z + x)]
    fn type5(#[case] test_function: Type5Function, #[case] target_function: fn(f32, f32, f32) -> f32) {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();
        let mut data3 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();
            let array3: Array<1> = data3.clone().into();

            let result: Vec<f32> = test_function(&array1, &array2, &array3).into();

            for (((d1, d2), d3), r) in data1.iter().zip(data2.iter()).zip(data3.iter()).zip(result.iter()) {
                assert_eq!(*r, target_function(*d1, *d2, *d3));
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
            data3.push((i + 2) as f32);
        }
    }

    #[rstest]
    #[case::fmadd(Array::<1>::fmadd)]
    #[should_panic]
    fn type5_shape_mismatch(#[case] test_function: Type5Function) {
        let a: Array<1> = vec![0.0; 3].into();
        let b: Array<1> = vec![0.0; 4].into();
        let c: Array<1> = vec![0.0; 5].into();

        test_function(&a, &b, &c);
    }

    #[rstest]
    #[case::fmadd(Array::<1>::fmadd_in_place, |x, y, z| y * z + x)]
    fn fmadd_in_place(#[case] test_function: Type6Function, #[case] target_function: fn(f32, f32, f32) -> f32) {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();
        let mut data3 = Vec::new();

        for i in 0..64 {
            let mut array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();
            let array3: Array<1> = data3.clone().into();

            test_function(&mut array1, &array2, &array3);
            let result: Vec<f32> = array1.into();

            for (((d1, d2), d3), r) in data1.iter().zip(data2.iter()).zip(data3.iter()).zip(result.iter()) {
                assert_eq!(*r, target_function(*d1, *d2, *d3));
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
            data3.push((i + 2) as f32);
        }
    }

    #[rstest]
    #[case::fmadd(Array::<1>::fmadd_in_place)]
    #[should_panic]
    fn fmadd_in_place_shape_mismatch(#[case] test_function: Type6Function) {
        let mut a: Array<1> = vec![0.0; 3].into();
        let b: Array<1> = vec![0.0; 4].into();
        let c: Array<1> = vec![0.0; 5].into();       
        
        test_function(&mut a, &b, &c);
    }
}

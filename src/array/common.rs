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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_different_sizes() {
        let mut values = Vec::new();

        for i in 0..64 {
            let converted: Array<1> = values.clone().into();
            let converted_back: Vec<f32> = converted.into();

            assert_eq!(converted_back, values);

            values.push(i as f32);
        }
    }

    #[test]
    fn zeros_different_sizes() {
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

        for i in 0..128 {
            assert_eq!(array.get(i), data.get(i).map(|x| *x));
        }
    }

    #[test]
    fn set() {
        let mut data = Vec::new();

        for i in 0..64 {
            data.push(i as f32);
        }

        let mut array: Array<1> = data.clone().into();

        for i in 0..128 {
            let result = array.set(i, (i + 10) as f32);

            if i < 64 {
                assert_eq!(result, Some(()));
                assert_eq!(array.get(i), Some((i + 10) as f32));
            } else {
                assert_eq!(result, None);
            }
        }
    }

    #[test]
    fn add_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            let result: Vec<f32> = (array1 + array2).unwrap().into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1 + d2);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[test]
    fn add_shape_mismatch() {
        let array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let sum = array1 + array2;

        assert!(sum.is_err());
    }

    #[test]
    fn add_assign_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let mut array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            array1 += array2;
            let result: Vec<f32> = array1.into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1 + d2);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[test]
    #[should_panic]
    fn add_assign_shape_mismatch() {
        let mut array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        
        array1 += array2;
    }

    #[test]
    fn sub_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            let result: Vec<f32> = (array1 - array2).unwrap().into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1 - d2);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[test]
    fn sub_shape_mismatch() {
        let array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let result = array1 - array2;

        assert!(result.is_err());
    }

    #[test]
    fn sub_assign_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let mut array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            array1 -= array2;
            let result: Vec<f32> = array1.into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1 - d2);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[test]
    #[should_panic]
    fn sub_assign_shape_mismatch() {
        let mut array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();

        array1 -= array2;
    }

    #[test]
    fn mul_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            let result: Vec<f32> = (array1 * array2).unwrap().into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1 * d2);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[test]
    fn mul_shape_mismatch() {
        let array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let result = array1 * array2;

        assert!(result.is_err());
    }

    #[test]
    fn mul_assign_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let mut array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            array1 *= array2;
            let result: Vec<f32> = array1.into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1 * d2);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[test]
    #[should_panic]
    fn mul_assign_shape_mismatch() {
        let mut array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();

        array1 *= array2;
    }

    #[test]
    fn div_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            let result: Vec<f32> = (array1 / array2).unwrap().into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1 / d2);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[test]
    fn div_shape_mismatch() {
        let array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let result = array1 / array2;

        assert!(result.is_err());
    }

    #[test]
    fn div_assign_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let mut array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            array1 /= array2;
            let result: Vec<f32> = array1.into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1 / d2);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
        }
    }

    #[test]
    #[should_panic]
    fn div_assign_shape_mismatch() {
        let mut array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();

        array1 /= array2;
    }

    #[test]
    fn max_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            let result: Vec<f32> = array1.max(&array2).unwrap().into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1.max(*d2));
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);

            (data1, data2) = (data2, data1);
        }
    }

    #[test]
    fn max_shape_mismatch() {
        let array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let result = array1.max(&array2);

        assert!(result.is_err());
    }

    #[test]
    fn min_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();

            let result: Vec<f32> = array1.min(&array2).unwrap().into();

            for ((d1, d2), r) in data1.iter().zip(data2.iter()).zip(result.iter()) {
                assert_eq!(*r, d1.min(*d2));
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);

            (data1, data2) = (data2, data1);
        }
    }

    #[test]
    fn min_shape_mismatch() {
        let array1: Array<1> = vec![0.0; 3].into();
        let array2: Array<1> = vec![0.0; 4].into();
        let result = array1.min(&array2);

        assert!(result.is_err());
    }

    #[test]
    fn sqrt_different_sizes() {
        let mut data = Vec::new();

        for i in 0..64 {
            let array: Array<1> = data.clone().into();
            let result: Vec<f32> = array.sqrt().into();

            for (d, r) in data.iter().zip(result.iter()) {
                assert_eq!(*r, d.sqrt());
            }

            data.push(i as f32);
        }
    }

    #[test]
    fn fmadd_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();
        let mut data3 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();
            let array3: Array<1> = data3.clone().into();

            let result: Vec<f32> = array3.fmadd(&array1, &array2).unwrap().into();

            for (((d1, d2), d3), r) in data1.iter().zip(data2.iter()).zip(data3.iter()).zip(result.iter()) {
                assert_eq!(*r, *d1 * *d2 + d3);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
            data3.push((i + 2) as f32);
        }
    }

    #[test]
    fn fmadd_shape_mismatch() {
        let a: Array<1> = vec![0.0; 3].into();
        let b: Array<1> = vec![0.0; 4].into();
        let c: Array<1> = vec![0.0; 5].into();
        let result = c.fmadd(&a, &b);

        assert!(result.is_err());
    }

    #[test]
    fn fmadd_in_place_different_sizes() {
        let mut data1 = Vec::new();
        let mut data2 = Vec::new();
        let mut data3 = Vec::new();

        for i in 0..64 {
            let array1: Array<1> = data1.clone().into();
            let array2: Array<1> = data2.clone().into();
            let mut array3: Array<1> = data3.clone().into();

            array3.fmadd_in_place(&array1, &array2).unwrap();
            let result: Vec<f32> = array3.into();

            for (((d1, d2), d3), r) in data1.iter().zip(data2.iter()).zip(data3.iter()).zip(result.iter()) {
                assert_eq!(*r, *d1 * *d2 + d3);
            }

            data1.push(i as f32);
            data2.push((i + 1) as f32);
            data3.push((i + 2) as f32);
        }
    }

    #[test]
    fn fmadd_in_place_shape_mismatch() {
        let a: Array<1> = vec![0.0; 3].into();
        let b: Array<1> = vec![0.0; 4].into();
        let mut c: Array<1> = vec![0.0; 5].into();       
        let result = c.fmadd_in_place(&a, &b);

        assert!(result.is_err());
    }
}

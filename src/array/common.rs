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
    fn conversion_small() {
        let value = vec![0.0, 1.0, 2.0];
        let converted: Array<1> = value.clone().into();
        let converted_back: Vec<f32> = converted.into();

        assert_eq!(converted_back, value);
    }

    #[test]
    fn conversion_one_full_register() {
        let value = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let converted: Array<1> = value.clone().into();
        let converted_back: Vec<f32> = converted.into();

        assert_eq!(converted_back, value);
    }

    #[test]
    fn conversion_two_register() {
        let value = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let converted: Array<1> = value.clone().into();
        let converted_back: Vec<f32> = converted.into();

        assert_eq!(converted_back, value);
    }

    #[test]
    fn get() {
        let array: Array<1> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ]
        .into();

        assert_eq!(array.get(0), Some(0.0));
        assert_eq!(array.get(15), Some(15.0));
        assert_eq!(array.get(16), Some(16.0));
        assert_eq!(array.get(17), None);
    }

    #[test]
    fn set() {
        let mut array: Array<1> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ]
        .into();

        assert_eq!(array.set(0, 42.0), Some(()));
        assert_eq!(array.set(15, 37.5), Some(()));
        assert_eq!(array.set(16, 31.9), Some(()));
        assert_eq!(array.set(17, 95.4), None);

        let data: Vec<f32> = array.into();
        assert_eq!(
            data,
            vec![
                42.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                37.5, 31.9
            ]
        );
    }

    #[test]
    fn add_small() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0].into();
        let sum: Vec<f32> = (array1 + array2).unwrap().into();

        assert_eq!(sum, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn add_one_full_register() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]
        .into();
        let sum: Vec<f32> = (array1 + array2).unwrap().into();

        assert_eq!(
            sum,
            vec![
                3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
                31.0, 33.0
            ]
        );
    }

    #[test]
    fn add_two_registers() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0,
        ]
        .into();
        let sum: Vec<f32> = (array1 + array2).unwrap().into();

        assert_eq!(
            sum,
            vec![
                3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
                31.0, 33.0, 35.0
            ]
        );
    }

    #[test]
    fn add_shape_mismatch() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();
        let sum = array1 + array2;

        assert!(sum.is_err());
    }

    #[test]
    fn add_assign_small() {
        let mut array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0].into();

        array1 += array2;
        let sum: Vec<f32> = array1.into();

        assert_eq!(sum, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn add_assign_one_full_register() {
        let mut array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]
        .into();

        array1 += array2;

        let sum: Vec<f32> = array1.into();

        assert_eq!(
            sum,
            vec![
                3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
                31.0, 33.0
            ]
        );
    }

    #[test]
    fn add_assign_two_registers() {
        let mut array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0,
        ]
        .into();
        
        array1 += array2;

        let sum: Vec<f32> = array1.into();

        assert_eq!(
            sum,
            vec![
                3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
                31.0, 33.0, 35.0
            ]
        );
    }

    #[test]
    #[should_panic]
    fn add_assign_shape_mismatch() {
        let mut array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();
        
        array1 += array2;
    }

    #[test]
    fn sub_small() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0].into();
        let result: Vec<f32> = (array1 - array2).unwrap().into();

        assert_eq!(result, vec![-1.0; 3]);
    }

    #[test]
    fn sub_one_full_register() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]
        .into();
        let result: Vec<f32> = (array1 - array2).unwrap().into();

        assert_eq!(result, vec![-1.0; 16]);
    }

    #[test]
    fn sub_two_registers() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0,
        ]
        .into();
        let result: Vec<f32> = (array1 - array2).unwrap().into();

        assert_eq!(result, vec![-1.0; 17]);
    }

    #[test]
    fn sub_shape_mismatch() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();
        let result = array1 - array2;

        assert!(result.is_err());
    }

    #[test]
    fn sub_assign_small() {
        let mut array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0].into();

        array1 -= array2;

        let result: Vec<f32> = array1.into();

        assert_eq!(result, vec![-1.0; 3]);
    }

    #[test]
    fn sub_assign_one_full_register() {
        let mut array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]
        .into();
        
        array1 -= array2;

        let result: Vec<f32> = array1.into();

        assert_eq!(result, vec![-1.0; 16]);
    }

    #[test]
    fn sub_assign_two_registers() {
        let mut array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0,
        ]
        .into();
        
        array1 -= array2;

        let result: Vec<f32> = array1.into();

        assert_eq!(result, vec![-1.0; 17]);
    }

    #[test]
    #[should_panic]
    fn sub_assign_shape_mismatch() {
        let mut array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();

        array1 -= array2;
    }

    #[test]
    fn mul_small() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0].into();
        let result: Vec<f32> = (array1 * array2).unwrap().into();

        assert_eq!(result, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn mul_one_full_register() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]
        .into();
        let result: Vec<f32> = (array1 * array2).unwrap().into();

        assert_eq!(
            result,
            vec![
                2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0, 110.0, 132.0, 156.0, 182.0,
                210.0, 240.0, 272.0
            ]
        );
    }

    #[test]
    fn mul_two_registers() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0,
        ]
        .into();
        let result: Vec<f32> = (array1 * array2).unwrap().into();

        assert_eq!(
            result,
            vec![
                2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0, 110.0, 132.0, 156.0, 182.0,
                210.0, 240.0, 272.0, 306.0
            ]
        );
    }

    #[test]
    fn mul_shape_mismatch() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();
        let result = array1 * array2;

        assert!(result.is_err());
    }

    #[test]
    fn mul_assign_small() {
        let mut array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0].into();

        array1 *= array2;
        let result: Vec<f32> = array1.into();

        assert_eq!(result, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn mul_assign_one_full_register() {
        let mut array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]
        .into();
        
        array1 *= array2;
        let result: Vec<f32> = array1.into();

        assert_eq!(
            result,
            vec![
                2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0, 110.0, 132.0, 156.0, 182.0,
                210.0, 240.0, 272.0
            ]
        );
    }

    #[test]
    fn mul_assign_two_registers() {
        let mut array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0,
        ]
        .into();
        
        array1 *= array2;
        let result: Vec<f32> = array1.into();

        assert_eq!(
            result,
            vec![
                2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0, 110.0, 132.0, 156.0, 182.0,
                210.0, 240.0, 272.0, 306.0
            ]
        );
    }

    #[test]
    #[should_panic]
    fn mul_assign_shape_mismatch() {
        let mut array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();

        array1 *= array2;
    }

    #[test]
    fn div_small() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0].into();
        let result: Vec<f32> = (array1 / array2).unwrap().into();

        assert_eq!(result, vec![1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0]);
    }

    #[test]
    fn div_one_full_register() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]
        .into();
        let result: Vec<f32> = (array1 / array2).unwrap().into();

        assert_eq!(
            result,
            vec![
                1.0 / 2.0,
                2.0 / 3.0,
                3.0 / 4.0,
                4.0 / 5.0,
                5.0 / 6.0,
                6.0 / 7.0,
                7.0 / 8.0,
                8.0 / 9.0,
                9.0 / 10.0,
                10.0 / 11.0,
                11.0 / 12.0,
                12.0 / 13.0,
                13.0 / 14.0,
                14.0 / 15.0,
                15.0 / 16.0,
                16.0 / 17.0
            ]
        );
    }

    #[test]
    fn div_two_registers() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0,
        ]
        .into();
        let result: Vec<f32> = (array1 / array2).unwrap().into();

        assert_eq!(
            result,
            vec![
                1.0 / 2.0,
                2.0 / 3.0,
                3.0 / 4.0,
                4.0 / 5.0,
                5.0 / 6.0,
                6.0 / 7.0,
                7.0 / 8.0,
                8.0 / 9.0,
                9.0 / 10.0,
                10.0 / 11.0,
                11.0 / 12.0,
                12.0 / 13.0,
                13.0 / 14.0,
                14.0 / 15.0,
                15.0 / 16.0,
                16.0 / 17.0,
                17.0 / 18.0
            ]
        );
    }

    #[test]
    fn div_shape_mismatch() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();
        let result = array1 / array2;

        assert!(result.is_err());
    }

    #[test]
    fn div_assign_small() {
        let mut array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0].into();

        array1 /= array2;
        let result: Vec<f32> = array1.into();

        assert_eq!(result, vec![1.0 / 2.0, 2.0 / 3.0, 3.0 / 4.0]);
    }

    #[test]
    fn div_assign_one_full_register() {
        let mut array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]
        .into();
        
        array1 /= array2;
        let result: Vec<f32> = array1.into();

        assert_eq!(
            result,
            vec![
                1.0 / 2.0,
                2.0 / 3.0,
                3.0 / 4.0,
                4.0 / 5.0,
                5.0 / 6.0,
                6.0 / 7.0,
                7.0 / 8.0,
                8.0 / 9.0,
                9.0 / 10.0,
                10.0 / 11.0,
                11.0 / 12.0,
                12.0 / 13.0,
                13.0 / 14.0,
                14.0 / 15.0,
                15.0 / 16.0,
                16.0 / 17.0
            ]
        );
    }

    #[test]
    fn div_assign_two_registers() {
        let mut array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
            18.0,
        ]
        .into();
        
        array1 /= array2;
        let result: Vec<f32> = array1.into();

        assert_eq!(
            result,
            vec![
                1.0 / 2.0,
                2.0 / 3.0,
                3.0 / 4.0,
                4.0 / 5.0,
                5.0 / 6.0,
                6.0 / 7.0,
                7.0 / 8.0,
                8.0 / 9.0,
                9.0 / 10.0,
                10.0 / 11.0,
                11.0 / 12.0,
                12.0 / 13.0,
                13.0 / 14.0,
                14.0 / 15.0,
                15.0 / 16.0,
                16.0 / 17.0,
                17.0 / 18.0
            ]
        );
    }

    #[test]
    #[should_panic]
    fn div_assign_shape_mismatch() {
        let mut array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();

        array1 /= array2;
    }

    #[test]
    fn max_small() {
        let array1: Array<1> = vec![1.0, 3.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 2.0, 4.0].into();
        let result: Vec<f32> = array1.max(&array2).unwrap().into();

        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn max_one_full_register() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 12.0, 13.0, 14.0, 15.0, 17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.0, 13.0, 14.0, 15.0, 16.0, 16.0,
        ]
        .into();
        let result: Vec<f32> = array1.max(&array2).unwrap().into();

        assert_eq!(
            result,
            vec![
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0,
            ]
        );
    }

    #[test]
    fn max_two_registers() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0, 13.0, 14.0, 15.0, 16.0,
            18.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 12.0, 14.0, 15.0, 16.0, 17.0,
            17.0,
        ]
        .into();
        let result: Vec<f32> = array1.max(&array2).unwrap().into();

        assert_eq!(
            result,
            vec![
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0, 18.0,
            ]
        );
    }

    #[test]
    fn max_shape_mismatch() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();
        let result = array1.max(&array2);

        assert!(result.is_err());
    }

    #[test]
    fn min_small() {
        let array1: Array<1> = vec![1.0, 3.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 2.0, 4.0].into();
        let result: Vec<f32> = array1.min(&array2).unwrap().into();

        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn min_one_full_register() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 12.0, 13.0, 14.0, 15.0, 17.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 11.0, 13.0, 14.0, 15.0, 16.0, 16.0,
        ]
        .into();
        let result: Vec<f32> = array1.min(&array2).unwrap().into();

        assert_eq!(
            result,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        );
    }

    #[test]
    fn min_two_registers() {
        let array1: Array<1> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0, 13.0, 14.0, 15.0, 16.0,
            18.0,
        ]
        .into();
        let array2: Array<1> = vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 12.0, 14.0, 15.0, 16.0, 17.0,
            17.0,
        ]
        .into();
        let result: Vec<f32> = array1.min(&array2).unwrap().into();

        assert_eq!(
            result,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0,
            ]
        );
    }

    #[test]
    fn min_shape_mismatch() {
        let array1: Array<1> = vec![1.0, 2.0, 3.0].into();
        let array2: Array<1> = vec![2.0, 3.0, 4.0, 5.0].into();
        let result = array1.min(&array2);

        assert!(result.is_err());
    }

    #[test]
    fn sqrt_small() {
        let mut data: Vec<f32> = Vec::with_capacity(3);
        let mut target: Vec<f32> = Vec::with_capacity(3);

        for i in 0..3 {
            data.push(i as f32);
            target.push((i as f32).sqrt());
        }

        let data: Array<1> = data.into();
        let result: Vec<f32> = data.sqrt().into();

        for (r, t) in result.iter().zip(target.iter()) {
            assert!((r - t).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn sqrt_one_full_register() {
        let mut data: Vec<f32> = Vec::with_capacity(16);
        let mut target: Vec<f32> = Vec::with_capacity(16);

        for i in 0..16 {
            data.push(i as f32);
            target.push((i as f32).sqrt());
        }

        let data: Array<1> = data.into();
        let result: Vec<f32> = data.sqrt().into();

        for (r, t) in result.iter().zip(target.iter()) {
            assert!((r - t).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn sqrt_two_registers() {
        let mut data: Vec<f32> = Vec::with_capacity(17);
        let mut target: Vec<f32> = Vec::with_capacity(17);

        for i in 0..17 {
            data.push(i as f32);
            target.push((i as f32).sqrt());
        }

        let data: Array<1> = data.into();
        let result: Vec<f32> = data.sqrt().into();

        for (r, t) in result.iter().zip(target.iter()) {
            assert!((r - t).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn fmadd_small() {
        let mut a: Vec<f32> = Vec::with_capacity(3);
        let mut b: Vec<f32> = Vec::with_capacity(3);
        let mut c: Vec<f32> = Vec::with_capacity(3);
        let mut target: Vec<f32> = Vec::with_capacity(3);

        for i in 0..3 {
            a.push(i as f32);
            b.push((i + 1) as f32);
            c.push((i + 2) as f32);
            target.push(i as f32 * (i + 1) as f32 + (i + 2) as f32);
        }

        let a: Array<1> = a.into();
        let b: Array<1> = b.into();
        let c: Array<1> = c.into();        
        let result: Vec<f32> = c.fmadd(&a, &b).unwrap().into();

        assert_eq!(result, target);
    }

    #[test]
    fn fmadd_one_full_register() {
        let mut a: Vec<f32> = Vec::with_capacity(16);
        let mut b: Vec<f32> = Vec::with_capacity(16);
        let mut c: Vec<f32> = Vec::with_capacity(16);
        let mut target: Vec<f32> = Vec::with_capacity(16);

        for i in 0..3 {
            a.push(i as f32);
            b.push((i + 1) as f32);
            c.push((i + 2) as f32);
            target.push(i as f32 * (i + 1) as f32 + (i + 2) as f32);
        }

        let a: Array<1> = a.into();
        let b: Array<1> = b.into();
        let c: Array<1> = c.into();        
        let result: Vec<f32> = c.fmadd(&a, &b).unwrap().into();

        assert_eq!(result, target);
    }

    #[test]
    fn fmadd_two_registers() {
        let mut a: Vec<f32> = Vec::with_capacity(17);
        let mut b: Vec<f32> = Vec::with_capacity(17);
        let mut c: Vec<f32> = Vec::with_capacity(17);
        let mut target: Vec<f32> = Vec::with_capacity(17);

        for i in 0..3 {
            a.push(i as f32);
            b.push((i + 1) as f32);
            c.push((i + 2) as f32);
            target.push(i as f32 * (i + 1) as f32 + (i + 2) as f32);
        }

        let a: Array<1> = a.into();
        let b: Array<1> = b.into();
        let c: Array<1> = c.into();        
        let result: Vec<f32> = c.fmadd(&a, &b).unwrap().into();

        assert_eq!(result, target);
    }

    #[test]
    fn fmadd_shape_mismatch() {
        let mut a: Vec<f32> = Vec::with_capacity(3);
        let mut b: Vec<f32> = Vec::with_capacity(4);
        let mut c: Vec<f32> = Vec::with_capacity(5);

        for i in 0..3 {
            a.push(i as f32);
            b.push((i + 1) as f32);
            c.push((i + 2) as f32);
        }

        b.push(4.0);
        c.push(5.0);
        c.push(6.0);

        let a: Array<1> = a.into();
        let b: Array<1> = b.into();
        let c: Array<1> = c.into();        
        let result = c.fmadd(&a, &b);

        assert!(result.is_err());
    }

    #[test]
    fn fmadd_in_place_small() {
        let mut a: Vec<f32> = Vec::with_capacity(3);
        let mut b: Vec<f32> = Vec::with_capacity(3);
        let mut c: Vec<f32> = Vec::with_capacity(3);
        let mut target: Vec<f32> = Vec::with_capacity(3);

        for i in 0..3 {
            a.push(i as f32);
            b.push((i + 1) as f32);
            c.push((i + 2) as f32);
            target.push(i as f32 * (i + 1) as f32 + (i + 2) as f32);
        }

        let a: Array<1> = a.into();
        let b: Array<1> = b.into();
        let mut c: Array<1> = c.into();        
        c.fmadd_in_place(&a, &b).unwrap();

        let result: Vec<f32> = c.into();

        assert_eq!(result, target);
    }

    #[test]
    fn fmadd_in_place_one_full_register() {
        let mut a: Vec<f32> = Vec::with_capacity(16);
        let mut b: Vec<f32> = Vec::with_capacity(16);
        let mut c: Vec<f32> = Vec::with_capacity(16);
        let mut target: Vec<f32> = Vec::with_capacity(16);

        for i in 0..3 {
            a.push(i as f32);
            b.push((i + 1) as f32);
            c.push((i + 2) as f32);
            target.push(i as f32 * (i + 1) as f32 + (i + 2) as f32);
        }

        let a: Array<1> = a.into();
        let b: Array<1> = b.into();
        let mut c: Array<1> = c.into();        
        c.fmadd_in_place(&a, &b).unwrap();

        let result: Vec<f32> = c.into();

        assert_eq!(result, target);
    }

    #[test]
    fn fmadd_in_place_two_registers() {
        let mut a: Vec<f32> = Vec::with_capacity(17);
        let mut b: Vec<f32> = Vec::with_capacity(17);
        let mut c: Vec<f32> = Vec::with_capacity(17);
        let mut target: Vec<f32> = Vec::with_capacity(17);

        for i in 0..3 {
            a.push(i as f32);
            b.push((i + 1) as f32);
            c.push((i + 2) as f32);
            target.push(i as f32 * (i + 1) as f32 + (i + 2) as f32);
        }

        let a: Array<1> = a.into();
        let b: Array<1> = b.into();
        let mut c: Array<1> = c.into();        
        c.fmadd_in_place(&a, &b).unwrap();

        let result: Vec<f32> = c.into();

        assert_eq!(result, target);
    }

    #[test]
    fn fmadd_in_place_shape_mismatch() {
        let mut a: Vec<f32> = Vec::with_capacity(3);
        let mut b: Vec<f32> = Vec::with_capacity(4);
        let mut c: Vec<f32> = Vec::with_capacity(5);

        for i in 0..3 {
            a.push(i as f32);
            b.push((i + 1) as f32);
            c.push((i + 2) as f32);
        }

        b.push(4.0);
        c.push(5.0);
        c.push(6.0);

        let a: Array<1> = a.into();
        let b: Array<1> = b.into();
        let mut c: Array<1> = c.into();        
        let result = c.fmadd_in_place(&a, &b);

        assert!(result.is_err());
    }
}

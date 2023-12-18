#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::__m512;

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
}

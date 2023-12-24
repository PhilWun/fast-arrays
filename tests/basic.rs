mod utils;

use fast_arrays::Array;
use utils::get_random_f32_vec;

#[test]
fn conversion() {
    for i in 0..64 {
        let data = get_random_f32_vec(0, i);
        let converted: Array<1> = data.clone().into();
        let converted_back: Vec<f32> = converted.into();

        assert_eq!(converted_back, data);
    }
}

#[test]
fn zeros() {
    for i in 0..64 {
        let zeros: Vec<f32> = Array::zeros(i).into();

        assert_eq!(zeros, vec![0.0; i]);
    }
}

#[test]
fn get() {
    let data = get_random_f32_vec(0, 64);
    let array: Array<1> = data.clone().into();

    for i in 0..64 {
        assert_eq!(array.get(i), *data.get(i).unwrap());
    }
}

#[test]
#[should_panic]
fn get_out_of_bounds() {
    let data = get_random_f32_vec(0, 64);
    let array: Array<1> = data.into();

    array.get(64);
}

#[test]
fn set() {
    let data = get_random_f32_vec(0, 64);
    let mut array: Array<1> = data.clone().into();

    for i in 0..64 {
        array.set(i, (i + 10) as f32);
        assert_eq!(array.get(i), (i + 10) as f32);
    }
}

#[test]
#[should_panic]
fn set_out_of_bounds() {
    let data = get_random_f32_vec(0, 64);
    let mut array: Array<1> = data.into();
    array.set(64, 42.0);
}

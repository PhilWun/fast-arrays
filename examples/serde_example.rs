use fast_arrays::{Array, Mask};

#[allow(dead_code)]
fn array_example() {
    let mut array = Array::<2>::zeros(&[3, 4]);

    for i in 0..3 {
        for j in 0..4 {
            array.set(i, j, (i * 4 + j) as f32);
        }
    }

    let json_str = serde_json::to_string_pretty(&array).unwrap();

    println!("{}", json_str);

    let de_array: Array<2> = serde_json::from_str(&json_str).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            println!("{}", de_array.get(i, j));
        }
    }

    println!("{:?}", de_array.get_shape())
}

fn mask_example() {
    let mut array = Array::<2>::zeros(&[3, 4]);

    for i in 0..3 {
        for j in 0..4 {
            array.set(i, j, rand::random());
        }
    }

    let masks = array.compare_scalar_less_than(0.5);

    for i in 0..3 {
        for j in 0..4 {
            println!("{}", masks.get(i, j));
        }
    }

    let json_str = serde_json::to_string_pretty(&masks).unwrap();

    println!("{}", json_str);

    let de_mask: Mask<2> = serde_json::from_str(&json_str).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            println!("{}", de_mask.get(i, j));
        }
    }

    println!("{:?}", de_mask.get_shape())
}

fn main() {
    mask_example();
}

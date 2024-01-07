use std::time::Instant;

use fast_arrays::Array;

fn main() {
    let len = 1<<10;
    let iterations = 1000;
    let mut array1 = Array::zeros(&[len]);
    let mut seed = [2; 16];

    let time1 = Instant::now();

    for _ in 0..iterations {
        seed = array1.random_uniform_in_place(seed);
    }

    let time2 = Instant::now();
    let duration = (time2 - time1).as_secs_f32();

    println!("{}", array1.get(0));
    println!("{} GB/s", ((len * iterations * 4) as f32 / 1_000_000_000.0) / duration);
}

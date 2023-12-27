use std::time::Instant;

use fast_arrays::Array;

fn main() {
    let len = 1024;
    let iterations = 10_000_000;
    let mut array1 = Array::zeros(len);
    let array2: Array<1> = vec![1.0; len].into();
    
    let time1 = Instant::now();

    for _ in 0..iterations {
        array1.add_in_place(&array2);
    }

    let time2 = Instant::now();

    println!("{}", array1.get(0));
    println!("{} ms", (time2 - time1).as_millis());
    println!("{} Gflops", ((len * iterations) as f32 / (time2 - time1).as_secs_f32()) / 1_000_000_000.0);
    println!("{} GB/s", ((len * iterations) as f32 / (time2 - time1).as_secs_f32()) / 1_000_000_000.0 * 8.0);
}

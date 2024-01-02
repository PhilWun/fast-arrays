/*
Copyright 2023 Philipp Wundrack

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use std::time::Instant;

use fast_arrays::Array;

fn main() {
    let len = 1024;
    let iterations = 10_000_000;
    let mut array1 = Array::zeros(&[len]);
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

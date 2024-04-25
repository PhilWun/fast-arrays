use fast_arrays::Array;

fn main() {
    let array = Array::random_uniform(&[128], Array::<1, ()>::random_seed());
    let view = array.view(16, 32);

    println!("{}", array.get(16));
    println!("{}", view.get(0));
    
}

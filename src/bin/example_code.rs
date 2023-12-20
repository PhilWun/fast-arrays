use fast_arrays::Array;

fn main() {
    let array1 = Array::<1>::zeros(3);
    let array2: Array<1> = vec![1.0, 2.0, 3.0].into();
    let sum = array1 + array2;

    println!("{}", sum.get(0));
}

use fast_arrays::Array1D;

fn main() {
    let array1 = Array1D::zeros(3);
    let array2: Array1D = vec![1.0, 2.0, 3.0].into();
    let sum = array1.add(&array2);

    println!("{}", sum.get(0));
}

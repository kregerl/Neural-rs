use crate::neural::{ActivationFunction, Layer, NeuralNetwork};
mod neural;

fn main() {
    // let nn = NeuralNetwork::new(vec![1, 2, 3, 4, 5], ActivationFunction::Relu);
    let nn = neural_network!(
        (784, 512, ActivationFunction::Relu),
        (512, 256, ActivationFunction::Relu),
        (256, 128, ActivationFunction::Relu),
        (128, 10, ActivationFunction::Sigmoid)
    );
    println!("{:?}", nn);
}

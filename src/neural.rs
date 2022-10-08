use std::cell::RefCell;
use std::f64::consts::E;
use rand::{distributions::Uniform, Rng};

#[macro_export]
macro_rules! neural_network {
    ( $(($input:literal, $output:literal, $enum:ident::$act:ident)),+ ) => {
        {
            let mut tmp = Vec::new();
            $(
                tmp.push(Layer::new($input, $output, $enum::$act));
            )*
            NeuralNetwork::new(tmp)
        }
    };
}

#[derive(Debug)]
pub struct Layer {
    num_input_nodes: usize,
    num_output_nodes: usize,
    weight_cost_gradient: Vec<Vec<f64>>,
    weights: Vec<Vec<f64>>,
    bais_cost_gradient: Vec<f64>,
    biases: Vec<f64>,
    activation_function: ActivationFunction,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationFunction {
    Linear,
    Step,
    Sigmoid,
    Relu,
    Custom(fn(f64) -> f64),
}

#[derive(Debug)]
pub struct NeuralNetwork {
    layers: RefCell<Vec<Layer>>,
}

#[derive(Debug)]
pub struct DataPoint {
    inputs: Vec<f64>,
    expected_outputs: Vec<f64>,
}

impl Layer {
    pub fn new(num_input_nodes: usize, num_output_nodes: usize, activation_function: ActivationFunction) -> Self {
        Self {
            num_input_nodes,
            num_output_nodes,
            weight_cost_gradient: vec![vec![0.0; num_input_nodes]; num_output_nodes],
            weights: Layer::generate_random_weights(num_input_nodes, num_output_nodes),
            bais_cost_gradient: vec![0.0; num_output_nodes],
            biases: vec![0.0; num_output_nodes],
            activation_function,
        }
    }

    pub fn calculate_outputs(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut activations = vec![0.0; self.num_output_nodes];

        for node_out in 0..self.num_output_nodes {
            let mut weighted_input = self.biases[node_out];
            for node_in in 0..self.num_input_nodes {
                weighted_input += inputs[node_in] * self.weights[node_in][node_out];
            }
            activations[node_out] = self.activation_function(weighted_input);
        }
        activations
    }

    // Calculate loss of the single
    pub fn loss(output: f64, expected: f64) -> f64 {
        let error = expected - output;
        error * error
    }

    pub fn apply_gradients(&mut self, learn_rate: f64) {
        for node_out in 0..self.num_output_nodes {
            self.biases[node_out] -= self.bais_cost_gradient[node_out] * learn_rate;
            for node_in in 0..self.num_input_nodes {
                self.weights[node_in][node_out] = self.weight_cost_gradient[node_in][node_out] * learn_rate;
            }
        }
    }

    fn activation_function(&self, weighted_input: f64) -> f64 {
        match self.activation_function {
            ActivationFunction::Linear => { weighted_input }
            ActivationFunction::Step => { if weighted_input < 0.0 { 0.0 } else { 1.0 } }
            ActivationFunction::Sigmoid => { 1.0 / (1.0 + E.powf(-weighted_input)) }
            ActivationFunction::Relu => { f64::max(0.0, weighted_input) }
            ActivationFunction::Custom(f) => { f(weighted_input) }
        }
    }

    // Generate random weights in the range -1 to 1
    fn generate_random_weights(num_input_nodes: usize, num_output_nodes: usize) -> Vec<Vec<f64>> {
        let range = Uniform::from(-1.0..1.0);
        let mut weights = Vec::with_capacity(num_output_nodes);
        unsafe { weights.set_len(num_output_nodes); }
        for i in 0..num_output_nodes {
            let values: Vec<f64> = rand::thread_rng()
                .sample_iter(&range)
                .take(num_input_nodes)
                .map(|value| (value * 2.0 - 1.0) / (num_input_nodes as f64).sqrt())
                .collect();
            weights[i] = values;
        }
        weights
    }
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers: RefCell::new(layers)
        }
    }

    pub fn learn(&mut self, training_data: Vec<DataPoint>, learn_rate: f64) {
        let h = 0.0001;
        let original_loss = self.loss(training_data.as_slice());
        println!("Loss: {}", original_loss);

        for layer in &mut self.layers.take() {
            for node_in in 0..layer.num_input_nodes {
                for node_out in 0..layer.num_output_nodes {
                    // Calculate cost gradient for weights
                    layer.weights[node_in][node_out] += h;
                    let delta_cost = self.loss(training_data.as_slice()) - original_loss;
                    layer.weights[node_in][node_out] -= h;
                    layer.weight_cost_gradient[node_in][node_out] = delta_cost / h;
                }
            }

            for bias_idx in 0..layer.biases.len() {
                layer.biases[bias_idx] += h;
                let delta_cost: f64 = self.loss(training_data.as_slice()) - original_loss;
                layer.biases[bias_idx] -= h;
                layer.bais_cost_gradient[bias_idx] = delta_cost / h;
            }
        }
        self.apply_all_gradients(learn_rate);
    }

    // Calculated the outputs for given inputs
    pub fn calculate_outputs(&self, inputs: &[f64]) -> Vec<f64> {
        let mut result = inputs.to_vec();
        for layer in &self.layers.take() {
            result = layer.calculate_outputs(result);
        }
        result
    }

    // Retrieves the predicted value from the outputs (Max float in output layer)
    pub fn classify(&self, inputs: &[f64]) -> usize {
        let outputs = self.calculate_outputs(inputs);
        // Get max output index
        outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap()
    }

    fn apply_all_gradients(&self, learn_rate: f64) {
        for mut layer in self.layers.take() {
            layer.apply_gradients(learn_rate);
        }
    }

    // Calculate loss of a single datapoint
    fn datapoint_loss(&self, point: &DataPoint) -> f64 {
        let outputs = self.calculate_outputs(point.inputs.as_slice());
        let mut cost = 0.0;
        for i in 0..outputs.len() {
            cost += Layer::loss(outputs[i], point.expected_outputs[i]);
        }
        cost
    }

    // Calculate loss of all given datapoints
    fn loss(&self, points: &[DataPoint]) -> f64 {
        let size = points.len() as f64;
        let mut total_cost = 0.0;
        for point in points {
            total_cost += self.datapoint_loss(point);
        }
        total_cost / size
    }
}
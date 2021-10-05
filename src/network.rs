use ndarray::Array2;
use rand::Rng;

/// A simple neural network structure.
pub struct Network {
    dimension: Option<(usize, usize)>,
    /// Number of layers(depth) in this structure, including the input and output layer.
    num_layers: usize,
    /// The layer sizes vector.
    layers: Vec<usize>,
    /// Vector of biases, each is an Array2 of column size 1.
    biases: Vec<Array2<f64>>,
    /// Vector of weights.
    weights: Vec<Array2<f64>>,

    /// Vector of vector of filters
    kernel: Vec<Array2<f64>>,
    /// maxpooling size and sampling stride
    pooling: Option<(usize, usize)>,
}

impl Network {
    /// Train the network with stochastic gradient descent method.
    pub fn train(
        &mut self,
        trn_data: &[f64],
        trn_lbl: &[u8],
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
    ) {
        assert_eq!(trn_data.len(), trn_lbl.len() * self.layers[0]);
        assert_eq!(trn_lbl.len() % batch_size, 0);

        for era in 0..epochs {
            // TODO: shuffle dataset
            for i in 0..trn_lbl.len() / batch_size {
                self.train_batch(
                    &trn_data[i * self.layers[0] * batch_size
                        ..i * self.layers[0] * batch_size + self.layers[0] * batch_size],
                    &trn_lbl[i * batch_size..i * batch_size + batch_size],
                    learning_rate,
                );
            }
            println!("epoch {} complete", era);
        }
    }

    /// Train the network with given data in a batch.
    fn train_batch(&mut self, trn_data: &[f64], trn_lbl: &[u8], learning_rate: f64) {
        assert_eq!(trn_data.len(), trn_lbl.len() * self.layers[0]);

        let mut b_addup: Vec<Array2<f64>> = self
            .biases
            .iter()
            .map(|x| Array2::<f64>::zeros((*x).raw_dim()))
            .collect();

        let mut w_addup: Vec<Array2<f64>> = self
            .weights
            .iter()
            .map(|x| Array2::<f64>::zeros((*x).raw_dim()))
            .collect();

        for i in 0..trn_lbl.len() {
            let (delta_b, delta_w) = self.backward_propagate(
                &trn_data[i * self.layers[0]..i * self.layers[0] + self.layers[0]],
                trn_lbl[i],
            );

            assert_eq!(delta_b.len(), b_addup.len());
            assert_eq!(delta_w.len(), w_addup.len());

            b_addup
                .iter_mut()
                .zip(delta_b.iter().rev())
                .for_each(|(y, x)| *y += x);
            w_addup
                .iter_mut()
                .zip(delta_w.iter().rev())
                .for_each(|(y, x)| *y += x);
        }

        self.biases
            .iter_mut()
            .zip(b_addup.iter())
            .for_each(|(y, x)| *y -= &(x * learning_rate / trn_lbl.len() as f64));
        self.weights
            .iter_mut()
            .zip(w_addup.iter())
            .for_each(|(y, x)| *y -= &(x * learning_rate / trn_lbl.len() as f64));
    }

    /// From single data, compute gradient of parameters.
    /// Note that output is in reverse order of layer indexes.
    fn backward_propagate(
        &self,
        trn_data: &[f64],
        trn_lbl: u8,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        assert_eq!(trn_data.len(), self.layers[0]);

        let mut activation: Vec<Array2<f64>> =
            vec![Array2::from_shape_vec((self.layers[0], 1), trn_data.to_vec()).unwrap()];

        for i in 0..self.num_layers - 1 {
            let mut act = self.biases[i].clone() + &self.weights[i].dot(&activation[i]);
            act.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp())); // sigmoid
            activation.push(act);
        }

        let mut b_ret: Vec<Array2<f64>> = vec![];
        let mut w_ret: Vec<Array2<f64>> = vec![];

        let mut delta_h = activation[self.num_layers - 1].clone();
        delta_h[[trn_lbl as usize, 0]] -= 1.0; // TODO

        for i in (0..self.num_layers - 1).rev() {
            //will compute dE/dz = dh/dz * dE/dh
            let mut delta_z = delta_h.clone();
            delta_z
                .iter_mut()
                .zip(activation[i + 1].iter())
                .for_each(|(x, y)| *x *= (*y) * (1.0 - *y)); // sigmoid prime

            w_ret.push(delta_z.clone() * activation[i].t());

            delta_h = self.weights[i].t().dot(&delta_z);
            b_ret.push(delta_z);
        }

        (b_ret, w_ret)
    }

    /// Evaluate the network with a bunch of input data and labels
    pub fn evaluate(&self, tst_data: &[f64], tst_lbl: &[u8]) -> usize {
        assert_eq!(tst_data.len(), tst_lbl.len() * self.layers[0]);

        let mut answer = 0;
        for i in 0..tst_lbl.len() {
            let start = i * self.layers[0];
            if self.feed(&tst_data[start..start + self.layers[0]]) == tst_lbl[i] {
                answer += 1;
            }
        }

        answer
    }

    /// Infer the category of given data.
    fn feed(&self, tst_data: &[f64]) -> u8 {
        let mut v: Array2<f64> = if self.kernel.len() > 0 {
            // execute convolution here
            let dimension = self.dimension.unwrap();
            let v0 = Array2::<f64>::from_shape_vec((dimension.0, dimension.1), tst_data.to_vec())
                .unwrap();
            assert!(v0.shape()[0] >= self.kernel[0].shape()[0]);

            let mut features: Vec<Array2<f64>> = vec![];
            for kernel in self.kernel.iter() {
                let mut conv = Array2::<f64>::zeros((
                    dimension.0 - kernel.shape()[0] + 1,
                    dimension.1 - kernel.shape()[1] + 1,
                ));
                for i in 0..conv.shape()[0] {
                    for j in 0..conv.shape()[1] {
                        //TODO: do convolution here
                        conv[[i, j]] += 1.0;
                    }
                }
                features.push(conv);
            }

            let (pooling, stride) = self.pooling.unwrap();
            for item in features.iter() {
                let mut pooled = Array2::<f64>::from_elem((
                    item.shape()[0] - pooling + 1,
                    item.shape()[1] - pooling + 1,
                ), <f64>::MIN);

                for i in 0..pooled.shape()[0] {
                    for j in 0..pooled.shape()[1] {
                        for k in 0..pooling {
                            for l in 0..pooling {
                                if pooled[[i, j]] > item[[i + k, j + l]] {
                                    pooled[[i, j]] = item[[i + k, j + l]];
                                }
                            }
                        }
                    }
                }
            }

            //TODO: remove below, put flattened feature maps
            Array2::from_shape_vec((self.layers[0], 1), tst_data.to_vec()).unwrap()
        } else {
            assert_eq!(tst_data.len(), self.layers[0]);
            Array2::from_shape_vec((self.layers[0], 1), tst_data.to_vec()).unwrap()
        };

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            v = (*b).clone() + (*w).dot(&v);
            v.iter_mut().for_each(|x| *x = 1.0 / (1.0 + (-*x).exp()));
        }

        let argmax = v
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .unwrap();

        argmax.0 as u8
    }
}

/// Struct used for configuring the Network.
pub struct NetworkBuilder {
    /// A vector representing size of each layer.
    input_dim: Option<(usize, usize)>,
    layers: Vec<usize>,
    kernel_size: Option<(usize, usize)>,
    pooling_size: Option<(usize, usize)>,
}

impl NetworkBuilder {
    /// Create a new NetworkBuilder with no arguments set.
    pub fn new() -> NetworkBuilder {
        NetworkBuilder {
            input_dim: None,
            layers: vec![],
            kernel_size: None,
            pooling_size: None,
        }
    }
    /// Set the size of layers.
    pub fn set_layers(&mut self, arg: Vec<usize>) -> &mut NetworkBuilder {
        self.layers = arg;
        self
    }
    pub fn set_convolution(
        &mut self,
        dimension: (usize, usize),
        (kernel_size, num_of_filters, maxpool, subsample_stride): (usize, usize, usize, usize),
    ) -> &mut NetworkBuilder {
        assert!(maxpool > subsample_stride);
        self.input_dim = Some(dimension);
        self.kernel_size = Some((kernel_size, num_of_filters));
        self.pooling_size = Some((maxpool, subsample_stride));
        self
    }
    /// Get the Network instance with specified configuration.
    pub fn finalize(&mut self) -> Network {
        assert!(self.layers.len() > 1);
        let mut rng = rand::thread_rng();

        let mut kernel_vec: Vec<Array2<f64>> = vec![];

        match self.kernel_size {
            Some((kernel_size, num_of_filters)) => {
                let (rows, cols) = self.input_dim.unwrap();
                assert_eq!(self.layers[0], rows * cols);
                for _ in 0..num_of_filters {
                    kernel_vec.push(Array2::<f64>::ones((kernel_size, kernel_size)));
                }

                let (maxpool, stride) = self.pooling_size.unwrap();
                self.layers[0] -= kernel_size - 1;
                self.layers[0] -= maxpool - 1;
                self.layers[0] = (self.layers[0] + stride - 1) / stride;
                self.layers[0] *= num_of_filters
            }
            None => (),
        }

        let biases: Vec<Array2<f64>> = self
            .layers
            .iter()
            .skip(1)
            .map(|x| Array2::<f64>::from_shape_fn((*x, 1), |_| rng.gen_range(-1.0..1.0)))
            .collect();

        let weights: Vec<Array2<f64>> = self
            .layers
            .iter()
            .skip(1)
            .zip(self.layers.iter())
            .map(|(y, x)| Array2::<f64>::from_shape_fn((*y, *x), |_| rng.gen_range(-1.0..1.0)))
            .collect();

        assert_eq!(biases.len(), self.layers.len() - 1);
        assert_eq!(weights.len(), self.layers.len() - 1);

        Network {
            dimension: self.input_dim,
            num_layers: self.layers.len(),
            layers: self.layers.clone(), // TODO
            biases: biases,
            weights: weights,
            kernel: kernel_vec,
            pooling: self.pooling_size,
        }
    }
}

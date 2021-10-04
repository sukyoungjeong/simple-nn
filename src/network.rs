use ndarray::Array2;
use rand::Rng;

pub struct Network {
    num_layers: usize,
    layers: Vec<usize>,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
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

    fn feed(&self, tst_data: &[f64]) -> u8 {
        assert_eq!(tst_data.len(), self.layers[0]);
        let mut v: Array2<f64> =
            Array2::from_shape_vec((self.layers[0], 1), tst_data.to_vec()).unwrap();

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

pub struct NetworkBuilder {
    layers: Vec<usize>,
}

impl NetworkBuilder {
    pub fn new() -> NetworkBuilder {
        NetworkBuilder { layers: vec![] }
    }
    pub fn set_layers(&mut self, arg: Vec<usize>) -> &mut NetworkBuilder {
        self.layers = arg;
        self
    }
    pub fn finalize(&mut self) -> Network {
        assert!(self.layers.len() > 1);

        let mut rng = rand::thread_rng();

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
            num_layers: self.layers.len(),
            layers: self.layers.clone(), // TODO
            biases: biases,
            weights: weights,
        }
    }
}

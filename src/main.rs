extern crate mnist;
extern crate ndarray;
extern crate rand;

mod network;

use mnist::{Mnist, MnistBuilder};
use network::NetworkBuilder;

fn main() {
    let (trn_size, tst_size, rows, cols) = (50000usize, 10000usize, 28, 28);
    let (input_size, output_size) = (rows * cols as usize, 10usize);
    let (epochs, batch_size, lambda) = (3usize, 50usize, 3.0 as f64);

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .test_set_length(tst_size as u32)
        .download_and_extract()
        .finalize();

    let trn_img: Vec<f64> = trn_img.iter().map(|x| <f64>::from(*x) / 255.0).collect();
    let tst_img: Vec<f64> = tst_img.iter().map(|x| <f64>::from(*x) / 255.0).collect();

    let mut net = NetworkBuilder::new()
        .set_layers(vec![input_size, 16, 16, output_size])
        .finalize();

    net.train(
        trn_img.as_slice(),
        trn_lbl.as_slice(),
        epochs,
        batch_size,
        lambda,
    );

    println!(
        "Evaluation on test set: {} of {}",
        net.evaluate(tst_img.as_slice(), tst_lbl.as_slice()),
        tst_size
    );
}

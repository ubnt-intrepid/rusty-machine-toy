extern crate csv;
extern crate rand;
extern crate rustc_serialize;
extern crate rusty_machine as rm;

use rand::distributions::IndependentSample;
use rm::linalg::{Vector, Matrix};
use rm::learning::gmm::{CovOption, GaussianMixtureModel};
use rm::learning::UnSupModel;

#[allow(dead_code)]
#[derive(Debug, Clone, RustcDecodable)]
struct Iris {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: i32,
}

fn load_iris_dataset() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut reader = csv::Reader::from_file("./data/iris.csv").unwrap().has_headers(true);

    let mut dest = Vec::new();
    for row in reader.decode() {
        let row: Iris = row.unwrap();
        dest.push(vec![row.sepal_length, row.sepal_width]);
    }

    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Range::new(0.0, 1.0);

    dest.iter()
        .cloned()
        .partition(|_| dist.ind_sample(&mut rng) >= 0.02)
}

fn into_matrix(data: &Vec<Vec<f64>>, size: usize) -> Matrix<f64> {
    Matrix::new(data.len(),
                size,
                data.iter().flat_map(Clone::clone).collect::<Vec<f64>>())
}

fn main() {
    // read iris dataset.
    let (train_inputs, test_inputs) = load_iris_dataset();

    // construct Gaussian mixture model.
    let mut gmm = GaussianMixtureModel::new(2);
    gmm.set_max_iters(100);
    gmm.cov_option = CovOption::Full;

    // train.
    let inputs = into_matrix(&train_inputs, 2);
    gmm.train(&inputs).unwrap();

    // calculate
    let inputs = into_matrix(&test_inputs, 2);
    let probs = gmm.predict(&inputs).unwrap();

    //    plot_result(train_inputs,
    //                test_inputs,
    //                probs.into_vec(),
    //                gmm.means().cloned().unwrap(),
    //                gmm.covariances().cloned().unwrap(),
    //                gmm.mixture_weights().clone());
}
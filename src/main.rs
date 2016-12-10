extern crate csv;
extern crate rand;
extern crate rustc_serialize;
extern crate rusty_machine as rm;
extern crate rmp_serialize as msgpack;

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

fn load_iris() -> Vec<Vec<f64>> {
    let mut reader = csv::Reader::from_file("./data/iris.csv").unwrap().has_headers(true);

    let mut dest = Vec::new();
    for row in reader.decode() {
        let row: Iris = row.unwrap();
        dest.push(vec![row.sepal_length, row.sepal_width]);
    }
    dest
}

fn main() {
    // read iris dataset.
    let iris = load_iris();

    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Range::new(0.0, 1.0);

    let (train_inputs, test_inputs): (Vec<_>, Vec<_>) = iris.iter()
        .cloned()
        .partition(|_| dist.ind_sample(&mut rng) >= 0.02);

    let mut gmm = GaussianMixtureModel::new(2);
    gmm.set_max_iters(100);
    gmm.cov_option = CovOption::Full;

    {
        let inputs: Vec<f64> = train_inputs.iter().flat_map(Clone::clone).collect();
        let inputs = Matrix::new(train_inputs.len(), 2, inputs);
        gmm.train(&inputs).unwrap();
    }

    let probs;
    {
        let inputs: Vec<f64> = test_inputs.iter().flat_map(Clone::clone).collect();
        let inputs = Matrix::new(test_inputs.len(), 2, inputs);

        probs = gmm.predict(&inputs).unwrap().into_vec();
    }

    plot_result(train_inputs,
                test_inputs,
                probs,
                gmm.means().cloned().unwrap(),
                gmm.covariances().cloned().unwrap(),
                gmm.mixture_weights().clone());
}

fn plot_result(train_inputs: Vec<Vec<f64>>,
               test_inputs: Vec<Vec<f64>>,
               probs: Vec<f64>,
               means: Matrix<f64>,
               covariances: Vec<Matrix<f64>>,
               mixture_weights: Vector<f64>) {
    #[derive(RustcEncodable)]
    struct Value {
        train_inputs: Vec<Vec<f64>>,
        test_inputs: Vec<Vec<f64>>,
        probs: Vec<f64>,
        means: Vec<f64>,
        covariances: Vec<Vec<f64>>,
        mixture_weights: Vec<f64>,
    }
    let val = Value {
        train_inputs: train_inputs,
        test_inputs: test_inputs,
        probs: probs,
        means: means.into_vec(),
        covariances: covariances.into_iter().map(|c| c.into_vec()).collect(),
        mixture_weights: mixture_weights.into_vec(),
    };

    use rustc_serialize::Encodable;
    use msgpack::Encoder;
    let mut buf = Vec::new();
    val.encode(&mut Encoder::new(&mut buf)).unwrap();

    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("python")
        .arg("./scripts/plot.py")
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();

    child.stdin.as_mut().unwrap().write_all(&buf[..]).unwrap();

    child.wait_with_output().unwrap();
}

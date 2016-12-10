extern crate csv;
extern crate rand;
extern crate rustc_serialize;
extern crate rusty_machine as rm;
extern crate rmp_serialize as msgpack;

#[allow(dead_code)]
#[derive(Debug, Clone, RustcDecodable)]
struct Iris {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: i32,
}

fn load_iris() -> Vec<Iris> {
    let mut reader = csv::Reader::from_file("./data/iris.csv").unwrap().has_headers(true);

    let mut dest = Vec::new();
    for row in reader.decode() {
        let row: Iris = row.unwrap();
        dest.push(row);
    }
    dest
}


fn main() {
    // read iris dataset.
    let iris = load_iris();
    let n_datasets = ((iris.len() as f64) * 0.8) as usize;

    let trains: Vec<_> = iris.iter().take(n_datasets).cloned().collect();
    let tests: Vec<_> = iris.iter().skip(n_datasets).cloned().collect();
    drop(iris);

    use rm::linalg::{Vector, Matrix};

    let _inputs: Vec<f64> = trains.iter()
        .flat_map(|iris| vec![iris.petal_length, iris.petal_width])
        .collect();
    let inputs = Matrix::new(n_datasets, 2, _inputs);

    let _targets: Vec<f64> = trains.iter().map(|iris| iris.class.into()).collect();
    let targets = Vector::new(_targets);

    use rm::learning::logistic_reg::LogisticRegressor;
    use rm::learning::optim::grad_desc::GradientDesc;

    // create instances of optimizer and regressor.
    let gd = GradientDesc::default();
    let mut lr = LogisticRegressor::new(gd);

    use rm::learning::SupModel;
    lr.train(&inputs, &targets).unwrap();

    let lr = lr;


    let x1_test: Vec<f64> = tests.iter().map(|iris| iris.petal_length).collect();
    let x2_test: Vec<f64> = tests.iter().map(|iris| iris.petal_width).collect();

    let inputs: Vec<_> =
        x1_test.iter().zip(x2_test.iter()).flat_map(|(&x1, &x2)| vec![x1, x2]).collect();
    let inputs = Matrix::new(x1_test.len(), 2, inputs);

    let predicted = lr.predict(&inputs).unwrap();

    let params = lr.parameters().cloned().unwrap();

    plot_decision_regions(inputs, targets, params, x1_test, x2_test, predicted);
}

fn plot_decision_regions(inputs: rm::linalg::Matrix<f64>,
                         targets: rm::linalg::Vector<f64>,
                         params: rm::linalg::Vector<f64>,
                         x1_test: Vec<f64>,
                         x2_test: Vec<f64>,
                         predicted: rm::linalg::Vector<f64>) {
    #[derive(RustcEncodable)]
    struct Value {
        inputs: Vec<f64>,
        targets: Vec<f64>,
        params: Vec<f64>,
        x1_test: Vec<f64>,
        x2_test: Vec<f64>,
        predicted: Vec<f64>,
    }
    let val = Value {
        inputs: inputs.into_vec(),
        targets: targets.into_vec(),
        params: params.into_vec(),
        x1_test: x1_test,
        x2_test: x2_test,
        predicted: predicted.into_vec(),
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

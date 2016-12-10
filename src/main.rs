extern crate csv;
extern crate rustc_serialize;
extern crate rusty_machine as rm;
extern crate rmp_serialize as msgpack;

use std::{io, process};
use rm::linalg;

use std::io::Write;
use rm::learning::logistic_reg::LogisticRegressor;
use rm::learning::SupModel;

use rustc_serialize::Encodable;
use msgpack::Encoder;

#[allow(dead_code)]
#[derive(Debug,RustcDecodable)]
struct Iris {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: i32,
}

fn load_iris() -> io::Result<Vec<Iris>> {
    let mut reader = csv::Reader::from_file("./data/iris.csv").unwrap().has_headers(true);

    let mut dest = Vec::new();
    for row in reader.decode() {
        let row: Iris = row.unwrap();
        dest.push(row);
    }
    Ok(dest)
}

fn main() {
    let iris = load_iris().unwrap();
    let n_datasets = iris.len();
    let x1: Vec<_> = iris.iter().map(|iris| iris.petal_length).collect();
    let x2: Vec<_> = iris.iter().map(|iris| iris.petal_width).collect();
    let inputs: Vec<_> = x1.iter().zip(x2.iter()).flat_map(|(&x1, &x2)| vec![x1, x2]).collect();
    let inputs = linalg::Matrix::new(n_datasets, 2, inputs);
    let targets =
        linalg::Vector::new(iris.iter().map(|iris| iris.class.into()).collect::<Vec<f64>>());

    let mut lr = LogisticRegressor::default();
    lr.train(&inputs, &targets).unwrap();

    let val = (42u8, "the Answer");

    let mut buf = vec![0u8; 13];
    val.encode(&mut Encoder::new(&mut &mut buf[..])).unwrap();
    let mut child = process::Command::new("python")
        .arg("./scripts/plot.py")
        .stdin(process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(&buf[..]).unwrap();
    let output = child.wait_with_output().unwrap();
    drop(output);
}

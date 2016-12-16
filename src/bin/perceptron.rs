extern crate csv;
extern crate rustc_serialize;
#[macro_use]
extern crate itertools;
extern crate rustplotlib;

#[path = "../data/iris.rs"]
mod iris;

#[path = "../model/perceptron.rs"]
mod perceptron;

use iris::Iris;
use perceptron::Perceptron;
use rustplotlib::{Figure, Subplots, Axes2D, Scatter};
use rustplotlib::backend::{Backend, Matplotlib};

fn main() {
    let mut perceptron = Perceptron::default();

    let iris = Iris::from_file().expect("failed to read Iris dataset");
    let iris: Vec<_> = iris.into_iter().take(100).collect();

    let inputs = iris.iter().map(|ref iris| vec![iris.sepal_length, iris.petal_length]).collect();
    let targets = iris.iter()
        .map(|ref iris| match iris.label.as_str() {
            "Iris-setosa" => -1.0,
            _ => 1.0,
        })
        .collect();

    let errors = perceptron.fit(&inputs, &targets, 10);
    drop(errors);

    let x1: Vec<f64> = iris.iter().map(|ref iris| iris.sepal_length).collect();
    let x2: Vec<f64> = iris.iter().map(|ref iris| iris.petal_length).collect();
    let ax = Axes2D::default()
        .add(Scatter::new("setosa")
            .data(&x1[0..50], &x2[0..50])
            .marker("o")
            .color("red"))
        .add(Scatter::new("versicolor")
            .data(&x1[50..100], &x2[50..100])
            .marker("x")
            .color("blue"))
        .xlabel("Sepal Length")
        .ylabel("Petal Length")
        .grid(true);
    let fig = Figure::default().subplots(Subplots::new(1, 1).at(0, ax));

    let mut mpl = Matplotlib::new().unwrap();
    mpl.set_style("ggplot").unwrap();
    fig.apply(&mut mpl).unwrap();
    mpl.show().unwrap();
    mpl.wait().unwrap();
}

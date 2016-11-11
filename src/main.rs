extern crate rusty_machine;
extern crate rand;

use rusty_machine::linalg::{BaseMatrix, Matrix};
use rusty_machine::learning::k_means::KMeansClassifier;
use rusty_machine::learning::UnSupModel;

use rand::distributions::{Normal, IndependentSample};


fn generate_data(centroids: &Matrix<f64>, points_per_centroid: usize, noise: f64) -> Matrix<f64> {
    assert!(centroids.cols() > 0 && centroids.rows() > 0,
            "Centroids must not be empty.");
    assert!(noise >= 0.0, "Noise must be non-negative");

    let mut raw_cluster_data = Vec::with_capacity(centroids.rows() * points_per_centroid *
                                                  centroids.cols());

    let mut rng = rand::thread_rng();
    let normal_rv = Normal::new(0.0, noise);

    for _ in 0..points_per_centroid {
        for centroid in centroids.iter_rows() {
            let mut point = Vec::with_capacity(centroids.cols());
            for feature in centroid {
                point.push(feature + normal_rv.ind_sample(&mut rng));
            }

            raw_cluster_data.extend(point);
        }
    }

    Matrix::new(centroids.rows() * points_per_centroid,
                centroids.cols(),
                raw_cluster_data)
}

fn main() {
    // parameters:
    let samples_per_centroid = 2000;
    let centroids = Matrix::new(2, 2, vec![-0.5, -0.5, 0.0, 0.5]);
    let noise = 0.4;

    // generate samples
    let samples = generate_data(&centroids, samples_per_centroid, noise);

    // create a model with 2 clusters.
    let n_clusters = 2;
    let mut model = KMeansClassifier::new(n_clusters);

    // train the model.
    model.train(&samples).unwrap();

    let classes = model.predict(&samples).unwrap();
    let (first, second): (Vec<usize>, Vec<_>) = classes.data().iter().partition(|&x| *x == 0);

    println!("Samples closest to first centroid: {}", first.len());
    println!("Samples closest to second centroid: {}", second.len());
}

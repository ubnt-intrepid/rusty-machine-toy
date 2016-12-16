#[derive(Debug)]
pub struct Perceptron {
  eta: f64,
  w0: f64,
  w: Vec<f64>,
}

impl Perceptron {
  pub fn new(eta: f64) -> Perceptron {
    Perceptron {
      eta: eta,
      w0: 0.0,
      w: Vec::new(),
    }
  }

  /// fit to training data
  pub fn fit(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<f64>, n_iter: usize) -> Vec<usize> {
    let mut errors = Vec::new();

    self.w.resize(inputs[0].len(), 0.0);
    for _ in 0..n_iter {
      let mut err = 0;
      for (xi, yi) in izip!(inputs, targets) {
        let update = self.eta * (yi - self.predict(xi));
        self.w0 += update;
        for (mut w, x) in izip!(&mut self.w[..], &xi[..]) {
          *w += update * *x;
        }

        if update.abs() < 1e-12 {
          err += 1
        }
      }
      errors.push(err)
    }

    errors
  }

  fn net_input(&self, x: &Vec<f64>) -> f64 {
    izip!(x, &self.w).map(|(x, w)| x * w).sum::<f64>() + self.w0
  }

  fn predict(&self, x: &Vec<f64>) -> f64 {
    if self.net_input(x) >= 0.0 { 1.0 } else { -1.0 }
  }
}

impl Default for Perceptron {
  fn default() -> Perceptron {
    Perceptron::new(0.01)
  }
}

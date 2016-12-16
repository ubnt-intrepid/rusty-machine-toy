use csv;

const CONTENTS: &'static str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/iris.data"));

#[derive(Debug, RustcDecodable)]
pub struct Iris {
  pub sepal_length: f64,
  pub sepal_width: f64,
  pub petal_length: f64,
  pub petal_width: f64,
  pub label: String,
}

impl Iris {
  // https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
  pub fn from_file() -> Result<Vec<Iris>, csv::Error> {
    let mut reader = csv::Reader::from_string(CONTENTS).has_headers(false);

    let mut buf = Vec::new();
    for row in reader.decode() {
      let row = row?;
      buf.push(row);
    }

    Ok(buf)
  }
}

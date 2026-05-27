use pyo3::prelude::*;
use pyo3::types::PyModule;

/// A Python module implemented in Rust.
#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

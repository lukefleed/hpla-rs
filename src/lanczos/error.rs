//! Error types for Lanczos algorithm failures.

use thiserror::Error;

/// Represents all possible errors that can occur during a Lanczos process.
#[derive(Error, Debug)]
#[error(transparent)]
pub struct LanczosError(#[from] pub(crate) LanczosErrorKind);

#[derive(Error, Debug, PartialEq)]
pub(crate) enum LanczosErrorKind {
    /// The input vector `b` is numerically zero and cannot be normalized.
    #[error("input vector `b` must not be a zero vector")]
    ZeroInputVector,

    /// Dimensions of a parameter do not match the expected size.
    #[error("parameter mismatch: `{param_name}` expects size {expected}, but got {actual}")]
    ParameterMismatch {
        param_name: String,
        expected: usize,
        actual: usize,
    },

    /// A requested dimension exceeds the capacity pre-allocated in a
    /// workspace. Distinct from `ParameterMismatch` because the workspace
    /// value is an upper bound, not an equality constraint.
    #[error("capacity exceeded: `{param_name}` requested {requested}, but workspace was built with cap {cap}")]
    CapacityExceeded {
        param_name: String,
        cap: usize,
        requested: usize,
    },

    /// The user-provided f(T_k) solver returned an error.
    #[error("f(T_k) solver failed: {0}")]
    SolverError(String),
}

impl PartialEq for LanczosError {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_input_vector_error_message() {
        let error = LanczosError(LanczosErrorKind::ZeroInputVector);
        assert_eq!(
            error.to_string(),
            "input vector `b` must not be a zero vector"
        );
    }

    #[test]
    fn test_parameter_mismatch_error_message() {
        let error = LanczosError(LanczosErrorKind::ParameterMismatch {
            param_name: "y_k".to_string(),
            expected: 10,
            actual: 9,
        });
        assert_eq!(
            error.to_string(),
            "parameter mismatch: `y_k` expects size 10, but got 9"
        );
    }

    #[test]
    fn test_solver_error_message() {
        let error =
            LanczosError(LanczosErrorKind::SolverError("custom solver failed".to_string()));
        assert_eq!(
            error.to_string(),
            "f(T_k) solver failed: custom solver failed"
        );
    }
}

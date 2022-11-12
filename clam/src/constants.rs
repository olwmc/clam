//! Hard-coded constants for CLAM and its dependents.

// TODO: Maybe we let this be a user-specified parameter? Perhaps by an
// environment variable.
/// Subsample sqrt(n) instances if there are more than this many in a `Cluster`.
pub const SUB_SAMPLE_LIMIT: usize = 100;

/// For avoiding divide-by-zero errors.
pub const EPSILON: f64 = 1e-8;

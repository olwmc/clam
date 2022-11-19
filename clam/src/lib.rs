//! CLAM: Clustered Learning of Approximate Manifolds.

mod number;
mod cluster;
mod partition_criteria;

pub mod prelude;

pub mod constants;
pub mod helpers;

pub mod metric;
pub mod dataset;
pub mod space;

pub use number::Number;
pub use metric::Metric;
pub use dataset::Dataset;
pub use space::Space;
pub use partition_criteria::{PartitionCriterion, PartitionCriteria};
pub use cluster::{Cluster, Ratios};

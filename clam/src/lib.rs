//! CLAM: Clustered Learning of Approximate Manifolds.

mod number;
mod metric;
mod dataset;
mod space;
mod partition_criteria;
mod cluster;

pub mod prelude;
pub mod constants;
pub mod helpers;

pub use number::Number;
pub use metric::{
    Metric,
    Euclidean,
    EuclideanSq,
    Manhattan,
    Cosine,
    Hamming,
    Jaccard,
};
pub use dataset::{Dataset, Tabular};
pub use space::{Space, TabularSpace};
pub use partition_criteria::{
    PartitionCriterion,
    PartitionCriteria,
};
pub use cluster::{Cluster, Ratios};

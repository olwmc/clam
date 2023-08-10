//! CLAM-Accelerated K-nearest-neighbor Entropy-scaling Search.

pub mod codec;
pub mod knn;
pub mod rnn;

use distances::Number;
use rayon::prelude::*;

use crate::{Dataset, PartitionCriteria, Tree};

/// CLAM-Accelerated K-nearest-neighbor Entropy-scaling Search.
///
/// The search time scales by the metric entropy of the dataset.
///
/// # Type Parameters
///
/// * `T` - The type of the instances.
/// * `U` - The type of the distance value.
/// * `D` - The type of the dataset.
#[derive(Debug)]
pub struct Cakes<T: Send + Sync, U: Number, D: Dataset<T, U>> {
    /// The tree used for the search.
    tree: Tree<T, U, D>,
}

impl<T: Send + Sync, U: Number, D: Dataset<T, U>> Cakes<T, U, D> {
    /// Creates a new CAKES instance.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to search.
    /// * `seed` - The seed to use for the random number generator.
    /// * `criteria` - The criteria to use for partitioning the tree.
    #[allow(clippy::needless_pass_by_value)] // clippy is wrong in this case
    pub fn new(data: D, seed: Option<u64>, criteria: PartitionCriteria<T, U, D>) -> Self {
        Self {
            tree: Tree::new(data, seed).partition(&criteria),
        }
    }

    /// Returns a reference to the data.
    pub const fn data(&self) -> &D {
        self.tree.data()
    }

    /// Returns a reference to the tree.
    pub const fn tree(&self) -> &Tree<T, U, D> {
        &self.tree
    }

    /// Returns the depth of the tree.
    pub const fn depth(&self) -> usize {
        self.tree.depth()
    }

    /// Returns the radius of the root cluster of the tree.
    pub const fn radius(&self) -> U {
        self.tree.radius()
    }

    /// Performs a parallelized search for the nearest neighbors of a set of queries.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search around.
    /// * `radius` - The radius to search within.
    /// * `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of vectors of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn batch_rnn_search(&self, queries: &[T], radius: U, algorithm: rnn::Algorithm) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|query| self.rnn_search(query, radius, algorithm))
            .collect()
    }

    /// Performs a search for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search around.
    /// * `radius` - The radius to search within.
    /// * `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn rnn_search(&self, query: &T, radius: U, algorithm: rnn::Algorithm) -> Vec<(usize, U)> {
        algorithm.search(query, radius, &self.tree)
    }

    /// Performs a parallelized search for the nearest neighbors of a set of queries.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search around.
    /// * `k` - The number of neighbors to search for.
    /// * `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of vectors of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn batch_knn_search(&self, queries: &[T], k: usize, algorithm: knn::Algorithm) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|query| self.knn_search(query, k, algorithm))
            .collect()
    }

    /// Performs a search for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search around.
    /// * `k` - The number of neighbors to search for.
    /// * `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn knn_search(&self, query: &T, k: usize, algorithm: knn::Algorithm) -> Vec<(usize, U)> {
        algorithm.search(&self.tree, query, k)
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, collections::HashSet};

    use symagen::random_data;

    use crate::VecDataset;

    use super::*;

    fn metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        distances::vectors::euclidean(a, b)
    }

    #[test]
    #[ignore = "dataset updates are WIP"]
    fn test_search() {
        let data = vec![vec![0., 0.], vec![1., 1.], vec![2., 2.], vec![3., 3.]];

        let name = "test".to_string();
        let dataset = VecDataset::new(name, data, metric, false);
        let criteria = PartitionCriteria::new(true);
        let cakes = Cakes::new(dataset, None, criteria);

        let query = vec![0., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes
            .rnn_search(&query, 1.5, rnn::Algorithm::Clustered)
            .into_iter()
            .unzip();
        assert_eq!(results.len(), 2);

        let result_points = results
            .iter()
            .map(|&i| cakes.data().data[i].clone())
            .collect::<Vec<_>>();
        assert!(result_points.contains(&vec![0., 0.]));
        assert!(result_points.contains(&vec![1., 1.]));

        let query = vec![1., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes
            .rnn_search(&query, 0., rnn::Algorithm::Clustered)
            .into_iter()
            .unzip();
        assert_eq!(results.len(), 1);

        assert!(results
            .iter()
            .map(|&i| cakes.data().data[i].clone())
            .any(|x| x == [1., 1.].as_slice()));
    }

    #[test]
    #[ignore = "dataset updates are WIP"]
    fn rnn_search() {
        let data = (-100..=100).map(|x| vec![x.as_f32()]).collect::<Vec<_>>();
        let data = VecDataset::new("test".to_string(), data, metric, false);
        let criteria = PartitionCriteria::new(true);
        let cakes = Cakes::new(data, Some(42), criteria);

        let queries = (-10..=10).step_by(2).map(|x| vec![x.as_f32()]).collect::<Vec<_>>();
        for v in [2, 10, 50] {
            let radius = v.as_f32();
            let n_hits = 1 + 2 * v;

            for (i, query) in queries.iter().enumerate() {
                let linear_hits = {
                    let mut hits = cakes.rnn_search(query, radius, rnn::Algorithm::Linear);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };
                assert_eq!(
                    n_hits,
                    linear_hits.len(),
                    "Failed linear search: query: {i}, radius: {radius}, linear: {linear_hits:?}",
                );

                let ranged_hits = {
                    let mut hits = cakes.rnn_search(query, radius, rnn::Algorithm::Clustered);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };
                let linear_indices = linear_hits.iter().map(|&(i, _)| i).collect::<HashSet<_>>();
                let ranged_indices = ranged_hits.iter().map(|&(i, _)| i).collect::<HashSet<_>>();
                let diff = linear_indices.difference(&ranged_indices).copied().collect::<Vec<_>>();
                assert!(
                    diff.is_empty(),
                    "Failed Clustered search: query: {i}, radius: {radius}\nlnn: {linear_indices:?}\nrnn: {ranged_indices:?}\ndiff: {diff:?}",
                );
            }
        }
    }

    #[test]
    fn rnn_vectors() {
        let seed = 42;
        let (cardinality, dimensionality) = (10_000, 100);
        let (min_val, max_val) = (-1., 1.);

        let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);

        let num_queries = 100;
        let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed + 1);

        let name = format!("test-euclidean");
        let data = VecDataset::new(name, data.clone(), metric, false);
        let cakes = Cakes::new(data, Some(seed), PartitionCriteria::default());

        for radius in [0.0, 0.05, 0.1, 0.25, 0.5] {
            for (i, query) in queries.iter().enumerate() {
                let linear_hits = {
                    let mut hits = cakes.rnn_search(query, radius, rnn::Algorithm::Linear);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };

                let ranged_hits = {
                    let mut hits = cakes.rnn_search(query, radius, rnn::Algorithm::Clustered);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };
                let linear_indices = linear_hits.iter().map(|&(i, _)| i).collect::<HashSet<_>>();
                let ranged_indices = ranged_hits.iter().map(|&(i, _)| i).collect::<HashSet<_>>();
                let diff = linear_indices.difference(&ranged_indices).copied().collect::<Vec<_>>();
                assert!(
                    diff.is_empty(),
                    "Failed Clustered search: query: {i}, radius: {radius}\nlnn: {linear_indices:?}\nrnn: {ranged_indices:?}\ndiff: {diff:?}",
                );
            }
        }
    }

    #[test]
    fn rnn_strings() {
        let seed = 42;
        let (cardinality, alphabet) = (1_000, "ACTG");
        let (min_len, max_len) = (100, 100);

        let data = random_data::random_string(cardinality, min_len, max_len, alphabet, seed);

        let num_queries = 10;
        let queries = random_data::random_string(num_queries, min_len, max_len, alphabet, seed + 1);

        fn metric(a: &String, b: &String) -> u16 {
            distances::strings::levenshtein(a, b)
        }

        let name = format!("test-levenshtein");
        let data = VecDataset::new(name, data.clone(), metric, false);
        let cakes = Cakes::new(data, Some(42), PartitionCriteria::default());

        for radius in [1, 5, 10, 25] {
            for (i, query) in queries.iter().enumerate() {
                let linear_hits = {
                    let mut hits = cakes.rnn_search(query, radius, rnn::Algorithm::Linear);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };

                let ranged_hits = {
                    let mut hits = cakes.rnn_search(query, radius, rnn::Algorithm::Clustered);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };
                let linear_indices = linear_hits.iter().map(|&(i, _)| i).collect::<HashSet<_>>();
                let ranged_indices = ranged_hits.iter().map(|&(i, _)| i).collect::<HashSet<_>>();
                let diff = linear_indices.difference(&ranged_indices).copied().collect::<Vec<_>>();
                assert!(
                        diff.is_empty(),
                        "Failed Clustered search: query: {i}, radius: {radius}\nlnn: {linear_indices:?}\nrnn: {ranged_indices:?}\ndiff: {diff:?}",
                    );
            }
        }
    }
}

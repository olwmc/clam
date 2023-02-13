use rayon::prelude::*;

use crate::{prelude::*, utils::helpers};

#[derive(Debug)]
pub struct CAKES<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    space: &'a S,
    root: Cluster<'a, T, S>,
    depth: usize,
}

impl<'a, T, S> CAKES<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    pub fn new(space: &'a S) -> Self {
        CAKES {
            space,
            root: Cluster::new_root(space),
            depth: 0,
        }
    }

    pub fn build(mut self, criteria: &crate::PartitionCriteria<'a, T, S>) -> Self {
        self.root = self.root.partition(criteria, true);
        self.depth = self.root.max_leaf_depth();
        self
    }

    pub fn space(&self) -> &S {
        self.space
    }

    pub fn root(&self) -> &Cluster<'a, T, S> {
        &self.root
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn radius(&self) -> f64 {
        self.root.radius()
    }

    pub fn diameter(&self) -> f64 {
        self.root.radius() * 2.
    }

    #[inline(never)]
    pub fn batch_rnn_search(&self, queries_radii: &[(&[T], f64)]) -> Vec<Vec<(usize, f64)>> {
        queries_radii
            // .par_iter()
            .iter()
            .map(|(query, radius)| self.rnn_search(query, *radius))
            .collect()
    }

    pub fn rnn_search(&self, query: &[T], radius: f64) -> Vec<(usize, f64)> {
        if self.root.distance_to_query(query) > (self.root.radius() + radius) {
            vec![]
        } else {
            let mut confirmed = Vec::new();

            let mut candidate_clusters = vec![&self.root];
            while !candidate_clusters.is_empty() {
                // let depth = candidate_clusters.first().unwrap().depth();
                // log::info!("Rnn at depth {depth} with {} candidates ...", candidate_clusters.len());

                (confirmed, candidate_clusters) = candidate_clusters
                    .drain(..)
                    .flat_map(|c| c.overlapping_children(query, radius))
                    .partition(|c| c.is_leaf());
            }

            let mut straddlers;
            (confirmed, straddlers) = confirmed.drain(..).partition(|c| c.is_singleton());

            let hits = confirmed.drain(..).flat_map(|c| {
                let indices = c.indices();
                let d = self.space.query_to_one(query, indices[0]);
                indices.into_iter().map(move |i| (i, d))
            });

            let indices = straddlers.drain(..).flat_map(|c| c.indices()).collect();
            hits.chain(self.linear_search(query, radius, Some(indices)).drain(..))
                .collect()
        }
    }

    // pub fn batch_knn_search(&'a self, queries: &'a [&[T]], k: usize) -> Vec<Vec<usize>> {
    //     queries
    //         .par_iter()
    //         // .iter()
    //         .map(|&query| self.knn_search(query, k))
    //         .collect()
    // }

    // pub fn knn_search(&'a self, query: &'a [T], k: usize) -> Vec<usize> {
    //     if k > self.root.cardinality() {
    //         self.root.indices()
    //     } else {
    //         let mut sieve = super::KnnSieve::new(self.root.children().to_vec(), query, k);
    //         let mut counter = 0;

    //         while !sieve.is_refined {
    //             sieve = sieve.refine_step(counter);
    //             counter += 1;
    //         }
    //         sieve.refined_extract()
    //     }
    // }

    pub fn batch_knn_by_rnn(&'a self, queries: &[&[T]], k: usize) -> Vec<Vec<(usize, f64)>> {
        queries
            // .par_iter()
            .iter()
            .map(|&q| self.knn_by_rnn(q, k))
            .collect()
    }

    pub fn knn_by_rnn(&'a self, query: &[T], k: usize) -> Vec<(usize, f64)> {
        let mut radius = self.root.radius() / self.root.cardinality().as_f64();
        let mut hits = self.rnn_search(query, radius);

        while hits.is_empty() {
            // TODO: Use EPSILON
            radius = radius * 2. + 1e-12;
            hits = self.rnn_search(query, radius);
        }

        while hits.len() < k {
            let distances = hits.iter().map(|(_, d)| *d).collect::<Vec<_>>();
            let lfd = helpers::compute_lfd(radius, &distances);
            let factor = ((k as f64) / (hits.len() as f64)).powf(1. / (lfd + 1e-12));
            assert!(factor > 1.);
            radius *= factor;
            hits = self.rnn_search(query, radius);
        }

        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        hits[..k].to_vec()
    }

    pub fn linear_search(&self, query: &[T], radius: f64, indices: Option<Vec<usize>>) -> Vec<(usize, f64)> {
        let indices = indices.unwrap_or_else(|| self.root.indices());
        let distances = self.space.query_to_many(query, &indices);
        indices
            .into_iter()
            .zip(distances.into_iter())
            .filter(|(_, d)| *d <= radius)
            .collect()
    }

    pub fn batch_linear_search(&self, queries_radii: &[(&[T], f64)]) -> Vec<Vec<(usize, f64)>> {
        queries_radii
            .par_iter()
            // .iter()
            .map(|(query, radius)| self.linear_search(query, *radius, None))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    use super::CAKES;

    #[test]
    #[ignore = "wip: non-binary trees"]
    fn test_search() {
        let data = vec![vec![0., 0.], vec![1., 1.], vec![2., 2.], vec![3., 3.]];
        let dataset = crate::Tabular::new(&data, "test_search".to_string());
        let metric = metric_from_name::<f64>("euclidean", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref());
        let cakes = CAKES::new(&space).build(&crate::PartitionCriteria::new(true));

        let query = &[0., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 1.5).into_iter().unzip();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));

        let query = cakes.space.data().get(1);
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 0.).into_iter().unzip();
        assert_eq!(results.len(), 1);
        assert!(!results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));
    }
}

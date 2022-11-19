use rayon::prelude::*;

use clam::prelude::*;

/// A `Vec` of 2-tuples of a `Cluster` and the distance from its center to the
/// query.
pub type ClusterResults<'a, T, U> = Vec<(&'a Cluster<'a, T, U>, U)>;

/// CAKES provides a hierarchical entropy-scaling search for ranged-nearest-
/// neighbors and k-nearest-neighbors.
///
/// If the `Metric` used is a distance metric, i.e. it obeys the triangle
/// inequality, then this search is exact.
#[derive(Debug, Clone)]
pub struct CAKES<'a, T: Number, U: Number> {
    space: &'a dyn Space<'a, T, U>,
    root: Cluster<'a, T, U>,
    depth: usize,
}

impl<'a, T: Number, U: Number> CAKES<'a, T, U> {
    pub fn new(space: &'a dyn Space<'a, T, U>) -> Self {
        CAKES {
            space,
            root: Cluster::new_root(space).build(),
            depth: 0,
        }
    }

    pub fn build(self, criteria: &PartitionCriteria<T, U>) -> Self {
        let root = self.root.partition(criteria, true);
        let depth = root.max_leaf_depth();
        CAKES {
            space: self.space,
            root,
            depth,
        }
    }

    pub fn space(&self) -> &dyn Space<'a, T, U> {
        self.space
    }

    pub fn data(&self) -> &dyn Dataset<'a, T> {
        self.space.data()
    }

    pub fn metric(&self) -> &dyn Metric<T, U> {
        self.space.metric()
    }

    pub fn root(&self) -> &Cluster<'a, T, U> {
        &self.root
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn radius(&self) -> U {
        self.root.radius()
    }

    pub fn diameter(&self) -> U {
        self.root.radius() * U::from(2).unwrap()
    }

    pub fn batch_rnn_search(&'a self, queries_radii: &[(&[T], U)]) -> Vec<Vec<(usize, U)>> {
        queries_radii
            .par_iter()
            // .iter()
            .map(|(query, radius)| self.rnn_search(query, *radius))
            .collect()
    }

    pub fn rnn_search(&'a self, query: &[T], radius: U) -> Vec<(usize, U)> {
        let [confirmed, straddlers] = self.rnn_tree_search(query, radius);

        let clusters = confirmed
            .into_iter()
            .map(|(c, _)| c)
            .chain(straddlers.into_iter().map(|(c, _)| c))
            .collect::<Vec<_>>();
        self.rnn_leaf_search(query, radius, &clusters)
    }

    pub fn rnn_tree_search(&'a self, query: &[T], radius: U) -> [ClusterResults<'a, T, U>; 2] {
        let mut confirmed = Vec::new();
        let mut straddlers = Vec::new();
        let mut candidate_clusters = vec![self.root()];

        while !candidate_clusters.is_empty() {
            let (terminal, non_terminal): (Vec<_>, Vec<_>) = candidate_clusters
                .into_iter()
                .map(|c| (c, self.space.query_to_one(query, c.arg_center())))
                .filter(|&(c, d)| d <= (c.radius() + radius))
                .partition(|&(c, d)| (c.radius() + d) <= radius);
            confirmed.extend(terminal.into_iter());

            let (terminal, non_terminal): (Vec<_>, Vec<_>) =
                non_terminal.into_iter().partition(|&(c, _)| c.is_leaf());
            straddlers.extend(terminal.into_iter());

            candidate_clusters = non_terminal
                .into_iter()
                .flat_map(|(c, _)| c.children())
                .collect();
        }

        [confirmed, straddlers]
    }

    pub fn rnn_leaf_search(
        &self,
        query: &[T],
        radius: U,
        candidate_clusters: &[&Cluster<T, U>],
    ) -> Vec<(usize, U)> {
        self.linear_search(
            query,
            radius,
            Some(
                candidate_clusters
                    .iter()
                    .flat_map(|&c| c.indices())
                    .collect(),
            ),
        )
    }

    pub fn linear_search(
        &self,
        query: &[T],
        radius: U,
        indices: Option<Vec<usize>>,
    ) -> Vec<(usize, U)> {
        let indices = indices.unwrap_or_else(|| self.root.indices());
        let distances = self.space.query_to_many(query, &indices);
        indices
            .into_iter()
            .zip(distances.into_iter())
            .filter(|(_, d)| *d <= radius)
            .collect()
    }

    pub fn batch_linear_search(&self, queries_radii: &[(Vec<T>, U)]) -> Vec<Vec<(usize, U)>> {
        queries_radii
            .par_iter()
            // .iter()
            .map(|(query, radius)| self.linear_search(query, *radius, None))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::CAKES;

    #[test]
    fn test_search() {
        let data = vec![vec![0., 0.], vec![1., 1.], vec![2., 2.], vec![3., 3.]];
        let dataset = clam::dataset::TabularDataset::new(&data, "test_search");
        let metric = clam::metric::cheap("euclidean");
        let space = clam::space::TabularSpace::<f64, f64>::new(&dataset, metric, false);
        let cakes = CAKES::new(&space).build(&clam::PartitionCriteria::new(true));

        let query = &[0., 1.];
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 1.5).into_iter().unzip();
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));

        let query = cakes.data().get(1);
        let (results, _): (Vec<_>, Vec<_>) = cakes.rnn_search(query, 0.).into_iter().unzip();
        assert_eq!(results.len(), 1);
        assert!(!results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
        assert!(!results.contains(&3));
    }
}

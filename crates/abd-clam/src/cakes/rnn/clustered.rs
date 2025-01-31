//! Clustered search for the ranged nearest neighbors of a query.

use distances::Number;

use crate::{Cluster, Dataset, Instance, Tree};

use super::linear;

/// Clustered search for the ranged nearest neighbors of a query.
///
/// # Arguments
///
/// * `tree` - The tree to search.
/// * `query` - The query to search around.
/// * `radius` - The radius to search within.
///
/// # Returns
///
/// A vector of 2-tuples, where the first element is the index of the instance
/// and the second element is the distance from the query to the instance.
pub fn search<I, U, D>(tree: &Tree<I, U, D>, query: &I, radius: U) -> Vec<(usize, U)>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
{
    let [confirmed, straddlers] = tree_search(tree.data(), &tree.root, query, radius);
    leaf_search(tree.data(), confirmed, straddlers, query, radius)
}

/// Perform coarse-grained tree search.
///
/// # Arguments
///
/// * `data` - The dataset to search.
/// * `root` - The root of the tree to search.
/// * `query` - The query to search around.
/// * `radius` - The radius to search within.
///
/// # Returns
///
/// A 2-slice of vectors of 2-tuples, where the first element in the slice
/// is the confirmed clusters, i.e. those that are contained within the
/// query ball, and the second element is the straddlers, i.e. those that
/// overlap the query ball. The 2-tuples are the clusters and the distance
/// from the query to the cluster center.
pub fn tree_search<'a, I, U, D>(data: &D, root: &'a Cluster<U>, query: &I, radius: U) -> [Vec<(&'a Cluster<U>, U)>; 2]
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
{
    let mut confirmed = Vec::new();
    let mut straddlers = Vec::new();
    let mut candidates = vec![root];

    let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
    while !candidates.is_empty() {
        (terminal, non_terminal) = candidates
            .into_iter()
            .map(|c| (c, c.distance_to_instance(data, query)))
            .filter(|&(c, d)| d <= (c.radius() + radius))
            .partition(|&(c, d)| (c.radius() + d) <= radius);
        confirmed.append(&mut terminal);

        (terminal, non_terminal) = non_terminal.into_iter().partition(|&(c, _)| c.is_leaf());
        straddlers.append(&mut terminal);

        candidates = non_terminal
            .into_iter()
            .flat_map(|(c, d)| {
                if d < c.radius() {
                    c.overlapping_children(data, query, radius)
                } else {
                    c.children()
                        .map_or_else(|| unreachable!("Non-leaf cluster without children"), |v| v.to_vec())
                }
            })
            .collect();
    }

    [confirmed, straddlers]
}

/// Perform fine-grained leaf search
pub fn leaf_search<I, U, D>(
    data: &D,
    confirmed: Vec<(&Cluster<U>, U)>,
    straddlers: Vec<(&Cluster<U>, U)>,
    query: &I,
    radius: U,
) -> Vec<(usize, U)>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
{
    let hits = confirmed.into_iter().flat_map(|(c, d)| {
        let distances = if c.is_singleton() {
            vec![d; c.cardinality()]
        } else {
            data.query_to_many(query, &c.indices().collect::<Vec<_>>())
        };
        c.indices().zip(distances)
    });

    let indices = straddlers
        .into_iter()
        .flat_map(|(c, _)| c.indices())
        .collect::<Vec<_>>();

    hits.chain(linear::search(data, query, radius, &indices)).collect()
}

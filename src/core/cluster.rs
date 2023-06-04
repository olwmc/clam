//! The `Cluster` is the heart of CLAM. It provides the ability to perform a
//! divisive hierarchical cluster of arbitrary datasets in arbitrary metric
//! spaces.

use std::hash::Hash;
use std::hash::Hasher;

use bitvec::prelude::*;

use super::cluster_criteria::PartitionCriteria;
use super::dataset::Dataset;
use super::number::Number;
use crate::utils::helpers;

pub type Ratios = [f64; 6];

/// A `Tree` is an abstraction over a `Cluster` hierarchy and their
/// associated data set.
///
/// Mirroring `Cluster`, typically one will chain calls to `new`, `build`,
/// and finally `partition` to construct a fully realized `Tree`.
#[derive(Debug)]
pub struct Tree<T: Number, U: Number, D: Dataset<T, U>> {
    data: D,
    root: Cluster<U>,
    t: std::marker::PhantomData<T>,
}

impl<T: Number, U: Number, D: Dataset<T, U>> Tree<T, U, D> {
    /// Constructs a new `Tree` for a given dataset. Importantly,
    /// this does not build nor partition the underlying `Cluster`
    /// tree.
    ///
    /// # Arguments
    /// dataset: A dataset to couple to the `Cluster` tree
    pub fn new(dataset: D) -> Self {
        Tree {
            root: Cluster::new_root(dataset.indices().to_owned()),
            data: dataset,
            t: std::marker::PhantomData::<T>,
        }
        // OWM: Should this call `build`, and `partition` by default?
    }

    /// # Returns
    /// A reference to the root `Cluster` of the tree
    pub(crate) fn root(&self) -> &Cluster<U> {
        &self.root
    }

    /// Returns a reference to dataset associated with the tree
    pub fn dataset(&self) -> &D {
        &self.data
    }

    /// # Returns
    /// The cardinality of the `Tree`'s root `Cluster`
    pub fn cardinality(&self) -> usize {
        self.root.cardinality()
    }

    /// # Returns
    /// The radius of the `Tree`'s root `Cluster`
    pub fn radius(&self) -> U {
        self.root.radius()
    }

    /// Sets a seed for the root cluster, returning a new `Tree` with
    /// the new seed associated with its root.
    ///
    /// # Arguments
    /// seed: A given seed
    ///
    /// # Returns
    /// A mutated tree instance with the new root having the given seed
    /// applied to it.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.root = self.root.with_seed(seed);
        self
    }

    /// Aliases `Cluster::par_partition` for the `Tree`'s root and dataset.
    ///
    /// # Arguments
    /// criteria: A `PartitionCriteria` through which the `Tree`'s root will be partitioned.
    ///
    /// # Returns
    /// A new `Tree` with a partitioned root.
    pub fn par_partition(mut self, criteria: &PartitionCriteria<U>, recursive: bool) -> Self {
        self.root = self.root.par_partition(&self.data, criteria, recursive);
        self
    }

    /// Aliases `Cluster::partition` for `Tree`'s root and dataset.
    ///
    /// # Arguments
    /// criteria: A `PartitionCriteria` through which the `Tree`'s root will be partitioned.
    ///
    /// # Returns
    /// A new `Tree` with a partitioned root.
    pub fn partition(mut self, criteria: &PartitionCriteria<U>, recursive: bool) -> Self {
        self.root = self.root.partition(&self.data, criteria, recursive);
        self
    }

    /// Aliases `Cluster::build` and builds the underlying root of the `Tree`.
    ///
    /// # Returns
    /// A `Tree` with a modified root that has gone through building.
    pub fn build(mut self) -> Self {
        self.root = self.root.build(&self.data);
        self
    }

    /// Returns the indices of the root cluster.
    pub fn indices(&self) -> &[usize] {
        self.root.indices(&self.data)
    }

    /// Reorders the `Tree`'s underlying dataset based off of a depth first traversal of a
    /// tree and reformats the `Cluster` hierarchy to use offset and cardinality based indices.
    pub fn depth_first_reorder(mut self) -> Self {
        // TODO (OWM): Is there a way to get around cloning here?
        let leaf_indices = self.root.leaf_indices();
        self.data.reorder(&leaf_indices);
        self.root.dfr(&self.data, 0);
        self
    }
}

/// A `Cluster` represents a collection of "similar" instances from a
/// metric-`Space`.
///
/// `Cluster`s can be unwieldy to use directly unless one has a good grasp of
/// the underlying invariants.
/// We anticipate that most users' needs will be well met by the higher-level
/// abstractions.
///
/// For most use-cases, one should chain calls to `new_root`, `build` and
/// `partition` to construct a tree on the metric space.
///
/// Clusters are named in the same way as nodes in a Huffman tree. The `root` is
/// named "1". A left child appends a "0" to the name of the parent and a right
/// child appends a "1".
///
/// For now, `Cluster` names are unique within a single tree. We plan on adding
/// tree-based prefixes which will make names unique across multiple trees.
#[derive(Debug)]
pub(crate) struct Cluster<U: Number> {
    cardinality: usize,
    history: BitVec,
    arg_center: usize,
    arg_radius: usize,
    radius: U,
    lfd: f64,
    ratios: Option<Ratios>,
    seed: Option<u64>,

    #[allow(clippy::type_complexity)]
    children: Option<([(usize, Box<Cluster<U>>); 2], U)>,
    index: Index,
}

#[derive(Debug)]
#[allow(dead_code)]
enum Index {
    // Leaf nodes only (Direct access)
    Indices(Vec<usize>),

    // All nodes after reordering (Direct access)
    Offset(usize),

    // Root nodes only (Indirect access through traversal )
    Empty,
}

impl<U: Number> PartialEq for Cluster<U> {
    fn eq(&self, other: &Self) -> bool {
        self.history == other.history
    }
}

/// Two clusters are equal if they have the same name. This only holds, for
/// now, for clusters in the same tree.
impl<U: Number> Eq for Cluster<U> {}

impl<U: Number> PartialOrd for Cluster<U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.depth() == other.depth() {
            self.history.partial_cmp(&other.history)
        } else {
            self.depth().partial_cmp(&other.depth())
        }
    }
}

/// `Cluster`s can be sorted based on their name. `Cluster`s are sorted by
/// non-decreasing depths and then by their names. Sorting a tree of `Cluster`s
/// will leave them in the order of a breadth-first traversal.
impl<U: Number> Ord for Cluster<U> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Clusters are hashed by their names. This means that a hash is only unique
/// within a single tree.
///
/// TODO: Add a tree-based prefix to the cluster names when we need to hash
/// clusters from different trees into the same collection.
impl<U: Number> Hash for Cluster<U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl<U: Number> std::fmt::Display for Cluster<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl<U: Number> Cluster<U> {
    /// Creates a new root `Cluster` for the metric space.
    ///
    /// # Arguments
    ///
    /// * `dataset`: on which to create the `Cluster`.
    pub fn new_root(indices: Vec<usize>) -> Self {
        Cluster::new(indices, bitvec![1])
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `dataset`: on which to create the `Cluster`.
    /// * `indices`: The indices of instances from the `dataset` that are
    /// contained in the `Cluster`.
    /// * `name`: `BitVec` name for the `Cluster`.
    pub fn new(indices: Vec<usize>, history: BitVec) -> Self {
        Cluster {
            cardinality: indices.len(),
            index: Index::Indices(indices),
            children: None,
            history,
            arg_center: 0,
            arg_radius: 0,
            radius: U::zero(),
            lfd: 0.0,
            ratios: None,
            seed: None,
        }
    }

    /// Computes and sets internal cluster properties including:
    /// - `arg_samples`
    /// - `arg_center`
    /// - `arg_radius`
    /// - `radius`
    /// - `lfd` (local fractal dimension)
    ///
    /// This method must be called before calling `partition` and before
    /// using the getter methods for those internal properties.
    pub fn build<T: Number, D: Dataset<T, U>>(mut self, data: &D) -> Self {
        let indices = match &self.index {
            Index::Indices(indices) => indices,
            // Behavior is analogous here. If we have an offset, then we have children
            Index::Offset(_) | Index::Empty => panic!("`build` can only be called once per cluster"),
        };

        // TODO: Explore with different values for the threshold e.g. 10, 100, 1000, etc.
        let arg_samples = if self.cardinality < 100 {
            indices.to_vec()
        } else {
            let n = ((indices.len() as f64).sqrt()) as usize;
            data.choose_unique(n, indices, self.seed)
        };

        let arg_center = data.median(&arg_samples);

        let center_distances = data.one_to_many(arg_center, indices);
        let (arg_radius, radius) = helpers::arg_max(&center_distances);
        let arg_radius = indices[arg_radius];

        self.arg_center = arg_center;
        self.arg_radius = arg_radius;
        self.radius = radius;
        self.lfd = helpers::compute_lfd(radius, &center_distances);

        self
    }

    /// Returns two new `Cluster`s that are the left and right children of this
    /// `Cluster`.
    fn partition_once<T: Number, D: Dataset<T, U>>(&self, data: &D) -> ([(usize, Self); 2], U) {
        let indices = match &self.index {
            Index::Indices(indices) => indices,
            _ => panic!("`build` can only be called once per cluster."),
        };

        let left_pole = self.arg_radius();
        let left_distances = data.one_to_many(left_pole, indices);

        let (arg_right, polar_distance) = helpers::arg_max(&left_distances);
        let right_pole = indices[arg_right];
        let right_distances = data.one_to_many(right_pole, indices);

        let (left, right) = indices
            .iter()
            .zip(left_distances.into_iter())
            .zip(right_distances.into_iter())
            .filter(|&((&i, _), _)| i != left_pole && i != right_pole)
            .partition::<Vec<_>, _>(|&((_, l), r)| l <= r);

        let left_indices = left
            .into_iter()
            .map(|((&i, _), _)| i)
            .chain([left_pole].into_iter())
            .collect::<Vec<_>>();
        let right_indices = right
            .into_iter()
            .map(|((&i, _), _)| i)
            .chain([right_pole].into_iter())
            .collect::<Vec<_>>();

        let (left_pole, left_indices, right_pole, right_indices) = if left_indices.len() < right_indices.len() {
            (right_pole, right_indices, left_pole, left_indices)
        } else {
            (left_pole, left_indices, right_pole, right_indices)
        };

        let left_name = {
            let mut name = self.history.clone();
            name.push(false);
            name
        };
        let right_name = {
            let mut name = self.history.clone();
            name.push(true);
            name
        };

        let left = Cluster::new(left_indices, left_name).build(data);
        let right = Cluster::new(right_indices, right_name).build(data);

        ([(left_pole, left), (right_pole, right)], polar_distance)
    }

    /// Partitions the `Cluster` based on the given criteria. If the `Cluster`
    /// can be partitioned, it will gain a pair of left and right child
    /// `Cluster`s. If called with the `recursive` flag, this will build the
    /// tree down to leaf `Cluster`s, i.e. `Cluster`s that can not be
    /// partitioned based on the given criteria.
    ///
    /// This method should be called after calling `build` and before calling
    /// the getter methods for children.
    ///
    /// # Arguments
    ///
    /// * `partition_criteria`: The rules by which to determine whether the
    /// cluster can be partitioned.
    /// * `recursive`: Whether to build the tree down to leaves using the same
    /// `partition_criteria`.
    ///
    /// # Panics:
    ///
    /// * If called before calling `build`.
    pub fn partition<T: Number, D: Dataset<T, U>>(
        mut self,
        data: &D,
        criteria: &PartitionCriteria<U>,
        recursive: bool,
    ) -> Self {
        if criteria.check(&self) {
            let ([(left_pole, left), (right_pole, right)], polar_distance) = self.partition_once(data);

            let (left, right) = if recursive {
                (
                    left.partition(data, criteria, recursive),
                    right.partition(data, criteria, recursive),
                )
            } else {
                (left, right)
            };

            self.children = Some((
                [(left_pole, Box::new(left)), (right_pole, Box::new(right))],
                polar_distance,
            ));
            self.index = Index::Empty;
        }
        self
    }

    #[allow(unused)]
    pub fn par_partition<T: Number, D: Dataset<T, U>>(
        mut self,
        data: &D,
        criteria: &PartitionCriteria<U>,
        recursive: bool,
    ) -> Self {
        if criteria.check(&self) {
            let ([(left_pole, left), (right_pole, right)], polar_distance) = self.partition_once(data);

            let (left, right) = if recursive {
                rayon::join(
                    || left.par_partition(data, criteria, recursive),
                    || right.par_partition(data, criteria, recursive),
                )
            } else {
                (left, right)
            };

            self.children = Some((
                [(left_pole, Box::new(left)), (right_pole, Box::new(right))],
                polar_distance,
            ));
            self.index = Index::Empty;
        }
        self
    }

    /// Computes and sets the `Ratios` for all `Cluster`s in the tree. These
    /// ratios are used for selecting `Graph`s for anomaly detection and other
    /// applications of CLAM.
    ///
    /// This method may only be called on a root cluster after calling the `build`
    /// and `partition` methods.
    ///
    /// # Arguments
    ///
    /// * `normalized`: Whether to normalize each ratio to a [0, 1] range based
    /// on the distribution of values for all `Cluster`s in the tree.
    ///
    /// # Panics:
    ///
    /// * If called on a non-root `Cluster`, i.e. a `Cluster` with depth > 0.
    /// * If called before `build` and `partition`.
    #[allow(unused_mut, unused_variables, dead_code)]
    pub fn with_ratios(mut self, normalized: bool) -> Self {
        todo!()
        // if !self.is_root() {
        //     panic!("This method may only be set from the root cluster.")
        // }
        // if self.is_leaf() {
        //     panic!("Please `build` and `partition` the tree before setting cluster ratios.")
        // }

        // match &self.index {
        //     Index::Indices(_) => panic!("Should not be here ..."),
        //     Index::Children(([(l, left), (r, right)], lr)) => {
        //         let left = Box::new(left.set_child_parent_ratios([1.; 6]));
        //         let right = Box::new(right.set_child_parent_ratios([1.; 6]));
        //         self.index = Index::Children(([(*l, left), (*r, right)], *lr));
        //     },
        // };
        // self.ratios = Some([1.; 6]);

        // if normalized {
        //     let ratios: Vec<_> = self.subtree().iter().flat_map(|c| c.ratios()).collect();
        //     let ratios: Vec<Vec<_>> = (0..6)
        //         .map(|s| ratios.iter().skip(s).step_by(6).cloned().collect())
        //         .collect();
        //     let means: [f64; 6] = ratios
        //         .iter()
        //         .map(|values| helpers::mean(values))
        //         .collect::<Vec<_>>()
        //         .try_into()
        //         .unwrap();
        //     let sds: [f64; 6] = ratios
        //         .iter()
        //         .zip(means.iter())
        //         .map(|(values, &mean)| 1e-8 + helpers::sd(values, mean))
        //         .collect::<Vec<_>>()
        //         .try_into()
        //         .unwrap();

        //     self.set_normalized_ratios(means, sds);
        // }

        // self
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn next_ema(&self, ratio: f64, parent_ema: f64) -> f64 {
        // TODO: Consider getting `alpha` from user. Perhaps via env vars?
        let alpha = 2. / 11.;
        alpha * ratio + (1. - alpha) * parent_ema
    }

    #[allow(unused_mut, unused_variables, dead_code)]
    fn set_child_parent_ratios(mut self, parent_ratios: Ratios) -> Self {
        todo!()
        // let [pc, pr, pl, pc_, pr_, pl_] = parent_ratios;

        // let c = (self.cardinality as f64) / pc;
        // let r = self.radius().as_f64() / pr;
        // let l = self.lfd() / pl;

        // let c_ = self.next_ema(c, pc_);
        // let r_ = self.next_ema(r, pr_);
        // let l_ = self.next_ema(l, pl_);

        // let ratios = [c, r, l, c_, r_, l_];
        // self.ratios = Some(ratios);

        // match &self.index {
        //     Index::Indices(_) => (),
        //     Index::Children(([(l, left), (r, right)], lr)) => {
        //         let left = Box::new(left.set_child_parent_ratios([1.; 6]));
        //         let right = Box::new(right.set_child_parent_ratios([1.; 6]));
        //         self.index = Index::Children(([(*l, left), (*r, right)], *lr));
        //     },
        // };

        // self
    }

    #[allow(unused_mut, unused_variables, dead_code)]
    fn set_normalized_ratios(&mut self, means: Ratios, sds: Ratios) {
        todo!()
        // let ratios: Vec<_> = self
        //     .ratios
        //     .unwrap()
        //     .into_iter()
        //     .zip(means.into_iter())
        //     .zip(sds.into_iter())
        //     .map(|((value, mean), std)| (value - mean) / (std * 2_f64.sqrt()))
        //     .map(libm::erf)
        //     .map(|v| (1. + v) / 2.)
        //     .collect();
        // self.ratios = Some(ratios.try_into().unwrap());

        // match self.index {
        //     Index::Indices(_) => (),
        //     Index::Children(([(_, mut left), (_, mut right)], _)) => {
        //         left.set_normalized_ratios(means, sds);
        //         right.set_normalized_ratios(means, sds);
        //     },
        // };
    }

    /// The number of instances in this `Cluster`.
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Returns the indices of the instances contained in this `Cluster`.
    ///
    /// Indices are only stored at leaf `Cluster`s. Calling this method on a
    /// non-leaf `Cluster` will have to perform a tree traversal, returning the
    /// indices in depth-first order.
    pub fn indices<'a, T: Number, D: Dataset<T, U>>(&'a self, data: &'a D) -> &[usize] {
        match &self.index {
            Index::Indices(indices) => indices,
            Index::Offset(o) => {
                let start = *o;
                &data.indices()[start..start + self.cardinality]
            }
            Index::Empty => panic!("Cannot call indices from parent clusters"),
        }
    }

    // OWM: Solely for depth first traversal
    pub fn leaf_indices(&self) -> Vec<usize> {
        match &self.index {
            Index::Empty => match &self.children {
                Some(([(_, left), (_, right)], _)) => left
                    .leaf_indices()
                    .iter()
                    .chain(right.leaf_indices().iter())
                    .copied()
                    .collect(),

                // TODO: Cleanup this error message
                None => panic!("Structural invariant invalidated. Node with no contents and no children"),
            },
            Index::Indices(indices) => indices.clone(),
            Index::Offset(_) => {
                panic!("Cannot get leaf indices once tree has been reordered!");
            }
        }
    }

    /// The `history` of the `Cluster` as a bool vector.
    #[allow(dead_code)]
    pub fn history(&self) -> Vec<bool> {
        self.history.iter().map(|v| *v).collect()
    }

    /// The `name` of the `Cluster` as a hex-String.
    pub fn name(&self) -> String {
        let d = self.history.len();
        let padding = if d % 4 == 0 { 0 } else { 4 - d % 4 };
        let bin_name = (0..padding)
            .map(|_| "0")
            .chain(self.history.iter().map(|b| if *b { "1" } else { "0" }))
            .collect::<Vec<_>>();
        bin_name
            .chunks_exact(4)
            .map(|s| {
                let [a, b, c, d] = [s[0], s[1], s[2], s[3]];
                let s = format!("{a}{b}{c}{d}");
                let s = u8::from_str_radix(&s, 2).unwrap();
                format!("{s:01x}")
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Whether the `Cluster` is the root of the tree.
    ///
    /// The root `Cluster` has a depth of 0.
    #[allow(dead_code)]
    pub fn is_root(&self) -> bool {
        self.depth() == 0
    }

    /// The number of parent-child hops from the root `Cluster` to this one.
    pub fn depth(&self) -> usize {
        self.history.len() - 1
    }

    /// The index of the instance at the center, i.e. the geometric median, of
    /// the `Cluster`.
    ///
    /// For `Cluster`s with a large `cardinality`, this is an approximation.
    ///
    /// TODO: Analyze the level of approximation for this. It's probably a
    /// sqrt(3) approximation based on some work in computational geometry.
    pub fn arg_center(&self) -> usize {
        self.arg_center
    }

    /// The index of the instance that is farthest from the `center`.
    pub fn arg_radius(&self) -> usize {
        self.arg_radius
    }

    /// The distance between the `center` and the instance farthest from the
    /// `center`.
    pub fn radius(&self) -> U {
        self.radius
    }

    /// Whether the `Cluster` contains only one instance or only identical
    /// instances.
    pub fn is_singleton(&self) -> bool {
        self.radius() == U::zero()
    }

    /// The local fractal dimension of the `Cluster` at the length scales of the
    /// `radius` and half that `radius`.
    #[allow(dead_code)]
    pub fn lfd(&self) -> f64 {
        self.lfd
    }

    #[allow(dead_code)]
    pub fn polar_distance(&self) -> Option<U> {
        self.children.as_ref().map(|(_, lr)| *lr)
    }

    /// The six `Cluster` ratios used for anomaly detection and related
    /// applications.
    ///
    /// These ratios are:
    ///
    /// * child-cardinality / parent-cardinality.
    /// * child-radius / parent-radius.
    /// * child-lfd / parent-lfd.
    /// * exponential moving average of child-cardinality / parent-cardinality.
    /// * exponential moving average of child-radius / parent-radius.
    /// * exponential moving average of child-lfd / parent-lfd.
    ///
    /// This method may only be called after calling `with_ratios` on the root.
    ///
    /// # Panics:
    ///
    /// * If called before calling `with_ratios` on the root.
    #[allow(dead_code)]
    pub fn ratios(&self) -> Ratios {
        self.ratios
            .expect("Please call `with_ratios` before using this method.")
    }

    /// A 2-slice of references to the left and right child `Cluster`s.
    pub fn children(&self) -> Option<[&Self; 2]> {
        self.children
            .as_ref()
            .map(|([(_, left), (_, right)], _)| [left.as_ref(), right.as_ref()])
    }

    /// Whether this cluster has no children.
    pub fn is_leaf(&self) -> bool {
        matches!(&self.index, Index::Indices(_))
    }

    /// Whether this `Cluster` is an ancestor of the `other` `Cluster`.
    #[allow(dead_code)]
    pub fn is_ancestor_of(&self, other: &Self) -> bool {
        self.depth() < other.depth() && self.history.iter().zip(other.history.iter()).all(|(l, r)| *l == *r)
    }

    /// Whether this `Cluster` is an descendant of the `other` `Cluster`.
    #[allow(dead_code)]
    pub fn is_descendant_of(&self, other: &Self) -> bool {
        other.is_ancestor_of(self)
    }

    /// A Vec of references to all `Cluster`s in the subtree of this `Cluster`,
    /// including this `Cluster`.
    pub fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];

        // Two scenarios: Either we have children or not
        match &self.children {
            Some(([(_, left), (_, right)], _)) => subtree
                .into_iter()
                .chain(left.subtree().into_iter())
                .chain(right.subtree().into_iter())
                .collect(),

            None => subtree,
        }
    }

    /// The number of descendants of this `Cluster`, excluding itself.
    #[allow(dead_code)]
    pub fn num_descendants(&self) -> usize {
        self.subtree().len() - 1
    }

    /// The maximum depth of any leaf in the subtree of this `Cluster`.
    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(|c| c.depth()).max().unwrap()
    }

    /// Distance from the `center` to the given indexed instance.
    #[allow(dead_code)]
    pub fn distance_to_indexed_instance<T: Number, D: Dataset<T, U>>(&self, data: &D, index: usize) -> U {
        data.one_to_one(index, self.arg_center())
    }

    /// Distance from the `center` to the given instance.
    pub fn distance_to_instance<T: Number, D: Dataset<T, U>>(&self, data: &D, instance: &[T]) -> U {
        data.query_to_one(instance, self.arg_center())
    }

    /// Distance from the `center` of this `Cluster` to the center of the
    /// `other` `Cluster`.
    #[allow(dead_code)]
    pub fn distance_to_other<T: Number, D: Dataset<T, U>>(&self, data: &D, other: &Self) -> U {
        self.distance_to_indexed_instance(data, other.arg_center())
    }

    /// Assuming that this `Cluster` overlaps with with query ball, we return
    /// only those children that also overlap with the query ball
    pub fn overlapping_children<T: Number, D: Dataset<T, U>>(&self, data: &D, query: &[T], radius: U) -> Vec<&Self> {
        let (l, left, r, right, lr) = match &self.children {
            None => panic!("Can only be called on non-leaf clusters."),
            Some(([(l, left), (r, right)], lr)) => (*l, left.as_ref(), *r, right.as_ref(), *lr),
        };
        let ql = data.query_to_one(query, l);
        let qr = data.query_to_one(query, r);

        let swap = ql < qr;
        let (ql, qr) = if swap { (qr, ql) } else { (ql, qr) };

        if (ql + qr) * (ql - qr) <= U::from(2).unwrap() * lr * radius {
            vec![left, right]
        } else if swap {
            vec![left]
        } else {
            vec![right]
        }
    }

    #[allow(dead_code)]
    // OWM: Do we need this anymore?
    pub fn depth_first_reorder<T: Number, D: Dataset<T, U>>(&mut self, data: &D) {
        if self.depth() != 0 {
            panic!("Cannot call this method except from the root.")
        }

        self.dfr(data, 0);
    }

    fn dfr<T: Number, D: Dataset<T, U>>(&mut self, data: &D, offset: usize) {
        self.index = Index::Offset(offset);

        // TODO: Cleanup
        self.arg_center = data.get_reordered_index(self.arg_center);
        self.arg_radius = data.get_reordered_index(self.arg_radius);

        if let Some(([(_, left), (_, right)], _)) = self.children.as_mut() {
            left.dfr(data, offset);
            right.dfr(data, offset + left.cardinality);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::cluster::{Cluster, Tree};
    use crate::core::cluster_criteria::PartitionCriteria;

    #[allow(unused_imports)]
    use crate::core::dataset::{Dataset, VecVec};
    use crate::distances;

    #[test]
    fn test_cluster() {
        let data = vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];
        let metric = distances::f32::euclidean;
        let name = "test".to_string();
        let data = VecVec::new(data, metric, name, false);
        let indices = data.indices().to_vec();
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
        let cluster = Cluster::new_root(indices)
            .build(&data)
            .partition(&data, &partition_criteria, true);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 4);
        assert_eq!(cluster.num_descendants(), 6);
        assert!(cluster.radius() > 0.);

        assert_eq!(format!("{cluster}"), "1");

        let [left, right] = cluster.children().unwrap();
        assert_eq!(format!("{left}"), "2");
        assert_eq!(format!("{right}"), "3");

        for child in [left, right] {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality(), 2);
            assert_eq!(child.num_descendants(), 2);
        }
    }

    #[test]
    fn test_reordering() {
        let data = vec![
            vec![10.],
            vec![1.],
            vec![-5.],
            vec![8.],
            vec![3.],
            vec![2.],
            vec![0.5],
            vec![0.],
        ];
        let metric = distances::f32::euclidean;
        let name = "test".to_string();
        let data = VecVec::new(data, metric, name, false);
        let partition_criteria: PartitionCriteria<f32> =
            PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

        let tree = Tree::new(data)
            .build()
            .partition(&partition_criteria, true)
            .depth_first_reorder();

        // Assert that the tree's indices have been reordered in depth-first order
        assert_eq!((0..tree.cardinality()).collect::<Vec<usize>>(), tree.indices());
    }

    #[test]
    fn test_leaf_indices() {
        let data = vec![
            vec![10.],
            vec![1.],
            vec![-5.],
            vec![8.],
            vec![3.],
            vec![2.],
            vec![0.5],
            vec![0.],
        ];
        let metric = distances::f32::euclidean;
        let name = "test".to_string();
        let data = VecVec::new(data, metric, name, false);
        let partition_criteria: PartitionCriteria<f32> =
            PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);

        let tree = Tree::new(data).build().partition(&partition_criteria, true);

        let mut leaf_indices = tree.root().leaf_indices();
        leaf_indices.sort();

        assert_eq!(leaf_indices, tree.dataset().indices());
    }
}

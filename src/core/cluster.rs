use std::f64::consts::SQRT_2;

use bitvec::prelude::*;

use super::partition_criteria::PartitionCriteria;
use crate::geometry::tetrahedron::*;
use crate::geometry::triangle::*;
use crate::prelude::*;
use crate::utils::helpers;

pub type Ratios = [f64; 6];
pub type History = BitVec;

/// Some values to be stored at build time that will be useful for partition.
type BuildCache = Vec<Vec<f64>>;

// TODO: Make this compatible with old geometry and search
#[derive(Debug)]
enum ClusterVariant {
    /// extrema
    Singleton([usize; 1]),
    /// diameter, radius, lfd, extrema
    Dipole(f64, f64, f64, [usize; 2]),
    /// radius, lfd, extrema, abc
    Trigon(f64, f64, [usize; 3], Triangle),
    /// radius, lfd, extrema, abc
    Tetragon(f64, f64, [usize; 4], Triangle),
}

impl ClusterVariant {
    fn radius(&self) -> f64 {
        match self {
            Self::Singleton(_) => 0.,
            Self::Dipole(_, r, ..) => *r,
            Self::Trigon(r, ..) => *r,
            Self::Tetragon(r, ..) => *r,
        }
    }

    fn lfd(&self) -> f64 {
        match self {
            Self::Singleton(_) => 1.,
            Self::Dipole(.., l, _) => *l,
            Self::Trigon(_, l, ..) => *l,
            Self::Tetragon(_, l, ..) => *l,
        }
    }

    fn extrema(&self) -> Vec<usize> {
        match self {
            Self::Singleton([a]) => vec![*a],
            Self::Dipole(.., [a, b]) => vec![*a, *b],
            Self::Trigon(.., [a, b, c], _) => vec![*a, *b, *c],
            Self::Tetragon(.., [a, b, c, d], _) => vec![*a, *b, *c, *d],
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::Singleton(_) => "Singleton",
            Self::Dipole(..) => "Dipole",
            Self::Trigon(..) => "Trigon",
            Self::Tetragon(..) => "Tetragon",
        }
    }
}

#[derive(Debug)]
pub enum Children<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    None(Vec<usize>),
    Dipole([Box<Cluster<'a, T, S>>; 2]),
    Trigon([Box<Cluster<'a, T, S>>; 3]),
    Tetragon([Box<Cluster<'a, T, S>>; 4]),
}

#[derive(Debug)]
pub struct Cluster<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    space: &'a S,
    history: History,
    cardinality: usize,
    variant: ClusterVariant,
    contents: Children<'a, T, S>,
    ratios: Option<Ratios>,
    t: std::marker::PhantomData<T>,
    naive_radius: f64,
    scaled_radius: f64,
    build_cache: Option<BuildCache>,
}

impl<'a, T, S> Cluster<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    pub fn new_root(space: &'a S) -> Self {
        let indices = space.data().indices();
        assert!(!indices.is_empty(), "`space.data().indices()` must not be empty.");
        Self::new(space, bitvec::bitvec![1], indices)
    }

    fn new(space: &'a S, history: History, indices: Vec<usize>) -> Self {
        match indices.len() {
            0 => panic!("Not allowed with no indices! {history:?}"),
            1 => Self::new_singleton(space, history, indices),
            2 => {
                let [a, b] = [indices[0], indices[1]];
                let diameter = space.one_to_one(a, b);
                let radius = diameter / 2.;
                Self::new_dipole(
                    space,
                    history,
                    vec![a, b],
                    diameter,
                    1.,
                    [a, b],
                    radius,
                    radius,
                    Some(vec![vec![0., diameter], vec![diameter, 0.]]), // TODO: Let this be None
                )
            }
            _ => {
                // The geometric median
                let m = {
                    let arg_samples = if indices.len() < 100 {
                        indices.clone()
                    } else {
                        let n = ((indices.len() as f64).sqrt()) as usize;
                        space.choose_unique(n, &indices)
                    };
                    let sample_distances = space
                        .pairwise(&arg_samples)
                        .into_iter()
                        .map(|v| v.into_iter().sum::<f64>())
                        .collect::<Vec<_>>();
                    arg_samples[helpers::arg_min(&sample_distances).0]
                };

                // the instance farthest from `m`
                let (a, naive_radius) = {
                    // TODO: Try profiling after doing this twice
                    let m_distances = space.one_to_many(m, &indices);
                    let (a, naive_radius) = helpers::arg_max(&m_distances);
                    (indices[a], naive_radius)
                };
                if naive_radius < EPSILON {
                    return Self::new_singleton(space, history, indices);
                }
                let a_distances = space.one_to_many(a, &indices);

                // the instance farthest from `a`
                let (b, ab) = {
                    let (b, ab) = helpers::arg_max(&a_distances);
                    (indices[b], ab)
                };
                let b_distances = space.one_to_many(b, &indices);

                // Make triangles to find `c`, the instance which makes the maximal cosine-angle with `a` and `b`.
                let triangles = indices
                    .iter()
                    .zip(a_distances.iter())
                    .zip(b_distances.iter())
                    .filter(|((&i, &ac), &bc)| {
                        i != a && i != b && ac > EPSILON && ab > EPSILON && makes_triangle([ab, ac, bc])
                    })
                    .map(|((&i, &ac), &bc)| (i, Triangle::with_edges_unchecked([ac, bc, ab])))
                    .collect::<Vec<_>>();

                if triangles.is_empty() {
                    // either there are only two unique instances or all instances are colinear
                    let radius = ab / 2.;
                    let radial_distances = a_distances.iter().map(|&d| (d - radius).abs()).collect::<Vec<_>>();
                    let lfd = helpers::compute_lfd(radius, &radial_distances);
                    return Self::new_dipole(
                        space,
                        history,
                        indices,
                        ab,
                        lfd,
                        [a, b],
                        radius,
                        radius,
                        Some(vec![a_distances, b_distances]),
                    );
                }

                // find `c` and the triangle `abc` which produced the maximal cosine
                let (c, cab) = triangles
                    .into_iter()
                    .max_by(|(_, l), (_, r)| l.cos_a().partial_cmp(&r.cos_a()).unwrap())
                    .unwrap();
                if cab.cos_a() <= 0. {
                    // No acute angle was possible so we have an ellipsoid-shaped Dipole. This should be rare.
                    let radius = ab / 2.;
                    let radial_distances = a_distances.iter().map(|&d| (d - radius).abs()).collect::<Vec<_>>();
                    let lfd = helpers::compute_lfd(radius, &radial_distances);
                    return Self::new_dipole(
                        space,
                        history,
                        indices,
                        ab,
                        lfd,
                        [a, b],
                        radius,
                        radius,
                        Some(vec![a_distances, b_distances]),
                    );
                }
                let c_distances = space.one_to_many(c, &indices);

                // TODO: Use `rotate` method on `Triangle`
                let [ac, bc, _] = cab.edge_lengths();
                let abc = Triangle::with_edges_unchecked([ab, ac, bc]);
                let triangle_radius = abc.r_sq().sqrt();

                // make tetrahedra to find the maximal radius for the cluster.
                let radial_distances = indices
                    .iter()
                    .zip(a_distances.iter())
                    .zip(b_distances.iter())
                    .zip(c_distances.iter())
                    .filter(|(((&i, &ad), &bd), &cd)| {
                        i != a
                            && i != b
                            && i != c
                            && ad > EPSILON
                            && bd > EPSILON
                            && cd > EPSILON
                            && makes_triangle([ab, ad, bd])
                            && makes_triangle([ac, ad, cd])
                            && makes_triangle([bc, bd, cd])
                    })
                    .map(|(((_, &ad), &bd), &cd)| {
                        Tetrahedron::with_edges_unchecked([ab, ac, bc, ad, bd, cd])
                            .od_sq()
                            .sqrt()
                    })
                    .collect::<Vec<_>>();

                if radial_distances.is_empty() {
                    Self::new_trigon(
                        space,
                        history,
                        indices,
                        triangle_radius,
                        2., // TODO: have a think about this case
                        [a, b, c],
                        abc,
                        naive_radius,
                        Some(vec![a_distances, b_distances, c_distances]),
                    )
                } else {
                    let (d, radius) = {
                        let (d, radius) = helpers::arg_max(&radial_distances);
                        (indices[d], radius)
                    };

                    if radius < triangle_radius {
                        let lfd = helpers::compute_lfd(triangle_radius, &radial_distances);
                        Self::new_trigon(
                            space,
                            history,
                            indices,
                            triangle_radius,
                            lfd,
                            [a, b, c],
                            abc,
                            naive_radius,
                            Some(vec![a_distances, b_distances, c_distances]),
                        )
                    } else {
                        let lfd = helpers::compute_lfd(radius, &radial_distances);
                        let d_distances = space.one_to_many(d, &indices);
                        Self::new_tetragon(
                            space,
                            history,
                            indices,
                            radius,
                            lfd,
                            [a, b, c, d],
                            abc,
                            naive_radius,
                            triangle_radius * SQRT_2,
                            Some(vec![a_distances, b_distances, c_distances, d_distances]),
                        )
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn new_singleton(space: &'a S, history: History, indices: Vec<usize>) -> Self {
        Self {
            space,
            history,
            cardinality: indices.len(),
            variant: ClusterVariant::Singleton([indices[0]]),
            contents: Children::None(indices),
            ratios: None,
            t: Default::default(),
            naive_radius: 0.,
            scaled_radius: 0.,
            build_cache: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn new_dipole(
        space: &'a S,
        history: History,
        indices: Vec<usize>,
        diameter: f64,
        lfd: f64,
        [a, b]: [usize; 2],
        naive_radius: f64,
        scaled_radius: f64,
        build_cache: Option<BuildCache>,
    ) -> Self {
        let radius = diameter / 2.;
        assert!(radius <= naive_radius, "radii: {radius:.12} vs {naive_radius:.12}");
        Self {
            space,
            history,
            cardinality: indices.len(),
            variant: ClusterVariant::Dipole(diameter, diameter / 2., lfd, [a, b]),
            contents: Children::None(indices),
            ratios: None,
            t: Default::default(),
            naive_radius,
            scaled_radius,
            build_cache,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn new_trigon(
        space: &'a S,
        history: History,
        indices: Vec<usize>,
        radius: f64,
        lfd: f64,
        [a, b, c]: [usize; 3],
        abc: Triangle,
        naive_radius: f64,
        build_cache: Option<BuildCache>,
    ) -> Self {
        // assert!(radius <= naive_radius, "radii: {radius:.12} vs {naive_radius:.12}");
        Self {
            space,
            history,
            cardinality: indices.len(),
            variant: ClusterVariant::Trigon(radius, lfd, [a, b, c], abc),
            contents: Children::None(indices),
            ratios: None,
            t: Default::default(),
            naive_radius,
            scaled_radius: radius,
            build_cache,
        }
    }

    #[allow(dead_code, clippy::too_many_arguments)]
    #[inline(always)]
    fn new_tetragon(
        space: &'a S,
        history: History,
        indices: Vec<usize>,
        radius: f64,
        lfd: f64,
        [a, b, c, d]: [usize; 4],
        abc: Triangle,
        naive_radius: f64,
        scaled_radius: f64,
        build_cache: Option<BuildCache>,
    ) -> Self {
        Self {
            space,
            history,
            cardinality: indices.len(),
            variant: ClusterVariant::Tetragon(radius, lfd, [a, b, c, d], abc),
            contents: Children::None(indices),
            ratios: None,
            t: Default::default(),
            naive_radius,
            scaled_radius,
            build_cache,
        }
    }

    pub fn partition(mut self, partition_criteria: &PartitionCriteria<'a, T, S>, recursive: bool) -> Self {
        if partition_criteria.check(&self) {
            match &self.variant {
                ClusterVariant::Singleton(..) => panic!("NOOOOOOOO!"),
                ClusterVariant::Dipole(..) => {
                    let [alpha, bravo] = self.partition_duo();
                    let (alpha, bravo) = if recursive {
                        // (
                        //     alpha.partition(partition_criteria, recursive),
                        //     bravo.partition(partition_criteria, recursive),
                        // )
                        rayon::join(
                            || alpha.partition(partition_criteria, recursive),
                            || bravo.partition(partition_criteria, recursive),
                        )
                    } else {
                        (alpha, bravo)
                    };
                    self.contents = Children::Dipole([Box::new(alpha), Box::new(bravo)]);
                }
                ClusterVariant::Trigon(..) => {
                    let [alpha, bravo, charlie] = self.partition_trio();
                    let ((alpha, bravo), charlie) = if recursive {
                        // (
                        //     (
                        //         alpha.partition(partition_criteria, recursive),
                        //         bravo.partition(partition_criteria, recursive),
                        //     ),
                        //     charlie.partition(partition_criteria, recursive),
                        // )
                        rayon::join(
                            || {
                                rayon::join(
                                    || alpha.partition(partition_criteria, recursive),
                                    || bravo.partition(partition_criteria, recursive),
                                )
                            },
                            || charlie.partition(partition_criteria, recursive),
                        )
                    } else {
                        ((alpha, bravo), charlie)
                    };
                    self.contents = Children::Trigon([Box::new(alpha), Box::new(bravo), Box::new(charlie)]);
                }
                ClusterVariant::Tetragon(..) => {
                    let [alpha, bravo, charlie, delta] = self.partition_quadro();
                    let (((alpha, bravo), charlie), delta) = if recursive {
                        // (
                        //     (
                        //         (
                        //             alpha.partition(partition_criteria, recursive),
                        //             bravo.partition(partition_criteria, recursive),
                        //         ),
                        //         charlie.partition(partition_criteria, recursive),
                        //     ),
                        //     delta.partition(partition_criteria, recursive),
                        // )
                        rayon::join(
                            || {
                                rayon::join(
                                    || {
                                        rayon::join(
                                            || alpha.partition(partition_criteria, recursive),
                                            || bravo.partition(partition_criteria, recursive),
                                        )
                                    },
                                    || charlie.partition(partition_criteria, recursive),
                                )
                            },
                            || delta.partition(partition_criteria, recursive),
                        )
                    } else {
                        (((alpha, bravo), charlie), delta)
                    };
                    self.contents =
                        Children::Tetragon([Box::new(alpha), Box::new(bravo), Box::new(charlie), Box::new(delta)]);
                }
            }
        }
        self.build_cache = None;
        self
    }

    fn partition_duo(&self) -> [Self; 2] {
        let indices = match &self.contents {
            Children::None(indices) => indices,
            _ => panic!("Impossible!"),
        };

        let extrema = self.extrema();
        let [a, b] = [extrema[0], extrema[1]];

        let [a_distances, b_distances] = {
            let build_cache = self.build_cache.as_ref().unwrap();
            [&build_cache[0], &build_cache[1]]
        };

        let (alpha, bravo) = {
            let (alpha, bravo): (Vec<_>, Vec<_>) = indices
                .iter()
                .zip(a_distances.iter().zip(b_distances.iter()))
                .filter(|(&i, _)| i != a && i != b)
                .partition(|(_, (&l, &r))| l <= r);

            let alpha = [a]
                .into_iter()
                .chain(alpha.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();
            let bravo = [b]
                .into_iter()
                .chain(bravo.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();

            if alpha.len() < bravo.len() {
                (bravo, alpha)
            } else {
                (alpha, bravo)
            }
        };

        let a_history = {
            let mut history = self.history.clone();
            history.push(false);
            history.push(false);
            history
        };
        let b_history = {
            let mut history = self.history.clone();
            history.push(false);
            history.push(true);
            history
        };

        [
            Self::new(self.space, a_history, alpha),
            Self::new(self.space, b_history, bravo),
        ]
    }

    fn partition_trio(&self) -> [Self; 3] {
        let indices = match &self.contents {
            Children::None(indices) => indices,
            _ => panic!("Impossible!"),
        };

        let extrema = self.extrema();
        let [a, b, c] = [extrema[0], extrema[1], extrema[2]];

        let [a_distances, b_distances, c_distances] = {
            let build_cache = self.build_cache.as_ref().unwrap();
            [&build_cache[0], &build_cache[1], &build_cache[2]]
        };

        let [alpha, bravo, charlie] = {
            let (alpha, bc): (Vec<_>, Vec<_>) = indices
                .iter()
                .zip(a_distances.iter().zip(b_distances.iter().zip(c_distances.iter())))
                .filter(|(&i, _)| i != a && i != b && i != c)
                .partition(|(_, (&a, (&b, &c)))| a <= b && a <= c);

            let (bravo, charlie): (Vec<_>, Vec<_>) = bc
                .into_iter()
                .filter(|(&i, _)| i != a && i != b && i != c)
                .partition(|(_, (_, (&b, &c)))| b <= c);

            let alpha = [a]
                .into_iter()
                .chain(alpha.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();
            let bravo = [b]
                .into_iter()
                .chain(bravo.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();
            let charlie = [c]
                .into_iter()
                .chain(charlie.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();

            let mut abc = [alpha, bravo, charlie];
            abc.sort_by_key(|a| -(a.len() as isize));
            abc
        };

        let a_history = {
            let mut history = self.history.clone();
            history.push(false);
            history.push(false);
            history
        };
        let b_history = {
            let mut history = self.history.clone();
            history.push(false);
            history.push(true);
            history
        };
        let c_history = {
            let mut history = self.history.clone();
            history.push(true);
            history.push(false);
            history
        };

        [
            Self::new(self.space, a_history, alpha),
            Self::new(self.space, b_history, bravo),
            Self::new(self.space, c_history, charlie),
        ]
    }

    fn partition_quadro(&self) -> [Self; 4] {
        let indices = match &self.contents {
            Children::None(indices) => indices,
            _ => panic!("Impossible!"),
        };

        let extrema = self.extrema();
        let [a, b, c, d] = [extrema[0], extrema[1], extrema[2], extrema[3]];

        let [a_distances, b_distances, c_distances, d_distances] = {
            let build_cache = self.build_cache.as_ref().unwrap();
            [&build_cache[0], &build_cache[1], &build_cache[2], &build_cache[3]]
        };

        let [alpha, bravo, charlie, delta] = {
            let (alpha, bcd): (Vec<_>, Vec<_>) = indices
                .iter()
                .zip(
                    a_distances
                        .iter()
                        .zip(b_distances.iter().zip(c_distances.iter().zip(d_distances.iter()))),
                )
                .filter(|(&i, _)| i != a && i != b && i != c && i != d)
                .partition(|(_, (&a, (&b, (&c, &d))))| a <= b && a <= c && a <= d);

            let (bravo, cd): (Vec<_>, Vec<_>) = bcd
                .into_iter()
                .filter(|(&i, _)| i != a && i != b && i != c && i != d)
                .partition(|(_, (_, (&b, (&c, &d))))| b <= c && b <= d);

            let (charlie, delta): (Vec<_>, Vec<_>) = cd
                .into_iter()
                .filter(|(&i, _)| i != a && i != b && i != c && i != d)
                .partition(|(_, (_, (_, (&c, &d))))| c <= d);

            let alpha = [a]
                .into_iter()
                .chain(alpha.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();
            let bravo = [b]
                .into_iter()
                .chain(bravo.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();
            let charlie = [c]
                .into_iter()
                .chain(charlie.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();
            let delta = [d]
                .into_iter()
                .chain(delta.into_iter().map(|(&i, _)| i))
                .collect::<Vec<_>>();

            let mut abcd = [alpha, bravo, charlie, delta];
            abcd.sort_by_key(|a| -(a.len() as isize));
            abcd
        };

        let a_history = {
            let mut history = self.history.clone();
            history.push(false);
            history.push(false);
            history
        };
        let b_history = {
            let mut history = self.history.clone();
            history.push(false);
            history.push(true);
            history
        };
        let c_history = {
            let mut history = self.history.clone();
            history.push(true);
            history.push(false);
            history
        };
        let d_history = {
            let mut history = self.history.clone();
            history.push(true);
            history.push(true);
            history
        };

        [
            Self::new(self.space, a_history, alpha),
            Self::new(self.space, b_history, bravo),
            Self::new(self.space, c_history, charlie),
            Self::new(self.space, d_history, delta),
        ]
    }

    #[allow(unused_mut, unused_variables)]
    pub fn with_ratios(mut self, normalized: bool) -> Self {
        // TODO:
        todo!()
    }

    pub fn space(&self) -> &'a S {
        self.space
    }

    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    pub fn indices(&self) -> Vec<usize> {
        match &self.contents {
            Children::None(indices) => indices.clone(),
            Children::Dipole(children) => children.iter().flat_map(|c| c.indices().into_iter()).collect(),
            Children::Trigon(children) => children.iter().flat_map(|c| c.indices().into_iter()).collect(),
            Children::Tetragon(children) => children.iter().flat_map(|c| c.indices().into_iter()).collect(),
        }
    }

    pub fn history(&self) -> &History {
        &self.history
    }

    // TODO: Optimize?
    pub fn name(&self) -> String {
        let d = self.history().len();
        let padding = 4 - d % 4;
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

    pub fn variant_name(&self) -> &str {
        self.variant.name()
    }

    pub fn depth(&self) -> usize {
        (self.history.len() - 1) / 2
    }

    pub fn is_singleton(&self) -> bool {
        matches!(self.variant, ClusterVariant::Singleton(..))
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self.contents, Children::None(..))
    }

    pub fn radius(&self) -> f64 {
        self.variant.radius()
    }

    pub fn naive_radius(&self) -> f64 {
        self.naive_radius
    }

    pub fn scaled_radius(&self) -> f64 {
        self.scaled_radius
    }

    pub fn lfd(&self) -> f64 {
        self.variant.lfd()
    }

    pub fn contents(&self) -> &Children<'a, T, S> {
        &self.contents
    }

    pub fn children(&self) -> Vec<&Self> {
        match &self.contents {
            Children::None(_) => panic!("Barren am I, ye' moron!"),
            Children::Dipole(children) => children.iter().map(|c| c.as_ref()).collect(),
            Children::Trigon(children) => children.iter().map(|c| c.as_ref()).collect(),
            Children::Tetragon(children) => children.iter().map(|c| c.as_ref()).collect(),
        }
    }

    pub fn ratios(&self) -> Ratios {
        // TODO:
        self.ratios
            .expect("Please call `with_ratios` after `build` before using this method.")
    }

    pub fn extrema(&self) -> Vec<usize> {
        self.variant.extrema()
    }

    pub fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];
        match &self.contents {
            Children::None(_) => subtree,
            Children::Dipole(children) => children.iter().flat_map(|c| c.subtree().into_iter()).collect(),
            Children::Trigon(children) => children.iter().flat_map(|c| c.subtree().into_iter()).collect(),
            Children::Tetragon(children) => children.iter().flat_map(|c| c.subtree().into_iter()).collect(),
        }
    }

    pub fn num_descendants(&self) -> usize {
        self.subtree().len() - 1
    }

    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(|c| c.depth()).max().unwrap()
    }

    pub fn distance_to_indexed(&self, index: usize) -> f64 {
        self.distance_to_query(self.space.data().get(index))
    }

    fn tetrahedral_distance(&self, extrema: [usize; 3], abc: &Triangle, query: &[T]) -> f64 {
        let distances = self.space.query_to_many(query, &extrema);
        let [ab, ac, bc] = abc.edge_lengths();
        let [ad, bd, cd] = [distances[0], distances[1], distances[2]];
        let distance = if let Ok(mut abcd) = Tetrahedron::with_edges(['a', 'b', 'c', 'd'], [ab, ac, bc, ad, bd, cd]) {
            abcd.od_sq()
        } else {
            abc.r_sq()
        };
        distance.sqrt()
    }

    pub fn distance_to_query(&self, query: &[T]) -> f64 {
        match &self.variant {
            ClusterVariant::Singleton([a]) => self.space.query_to_one(query, *a),
            ClusterVariant::Dipole(diameter, radius, _, [a, b]) => {
                let distances = self.space.query_to_many(query, &[*a, *b]);
                let [ac, bc] = [distances[0], distances[1]];
                if ac < EPSILON || bc < EPSILON {
                    *radius
                } else {
                    let abc = Triangle::with_edges_unchecked([*diameter, ac, bc]);
                    abc.cm_sq().sqrt()
                }
            }
            ClusterVariant::Trigon(.., [a, b, c], abc) => self.tetrahedral_distance([*a, *b, *c], abc, query),
            ClusterVariant::Tetragon(.., [a, b, c, _], abc) => self.tetrahedral_distance([*a, *b, *c], abc, query),
        }
    }

    pub fn distance_to_other(&self, other: &Self) -> f64 {
        match &self.variant {
            ClusterVariant::Singleton([a]) => {
                let center = self.space.data().get(*a);
                other.distance_to_query(center)
            }
            ClusterVariant::Dipole(diameter, radius, _, [a, b]) => {
                let [ac, bc] = [other.distance_to_indexed(*a), other.distance_to_indexed(*b)];
                if ac < EPSILON || bc < EPSILON {
                    *radius
                } else {
                    Triangle::with_edges_unchecked([*diameter, ac, bc]).cm_sq().sqrt()
                }
            }
            ClusterVariant::Trigon(.., [a, b, c], abc) => {
                let [ad, bd, cd] = [
                    other.distance_to_indexed(*a),
                    other.distance_to_indexed(*b),
                    other.distance_to_indexed(*c),
                ];
                let distance_sq = if ad < EPSILON || bd < EPSILON || cd < EPSILON {
                    abc.r_sq()
                } else {
                    Tetrahedron::with_triangle_unchecked(abc.clone(), [ad, bd, cd]).od_sq()
                };
                distance_sq.sqrt()
            }
            ClusterVariant::Tetragon(.., [a, b, c, _], abc) => {
                let [ad, bd, cd] = [
                    other.distance_to_indexed(*a),
                    other.distance_to_indexed(*b),
                    other.distance_to_indexed(*c),
                ];
                let distance_sq = if ad < EPSILON || bd < EPSILON || cd < EPSILON {
                    abc.r_sq()
                } else {
                    Tetrahedron::with_triangle_unchecked(abc.clone(), [ad, bd, cd]).od_sq()
                };
                distance_sq.sqrt()
            }
        }
    }
}

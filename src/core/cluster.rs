use bitvec::prelude::*;

use crate::geometry::triangle::*;
use crate::prelude::*;
use crate::utils::helpers;
use crate::PartitionCriteria;

pub type Ratios = [f64; 6];
type History = BitVec;
type BuildCache = [Vec<f64>; 2];

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
    contents: ClusterContents<'a, T, S>,
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
    pub fn new_root(space: &'a S, indices: Option<&[usize]>) -> Self {
        let indices = indices
            .map(|indices| {
                let max_i = helpers::arg_max(indices).1;
                assert!(
                    max_i < space.data().cardinality(),
                    "max of `indices` must be smaller than the cardinality of the dataset."
                );
                indices.to_vec()
            })
            .unwrap_or_else(|| space.data().indices());
        assert!(!indices.is_empty(), "`indices` must not be empty.");
        Self::new(space, bitvec::bitvec![1], indices)
    }

    fn new(space: &'a S, history: History, indices: Vec<usize>) -> Self {
        match indices.len() {
            0 => panic!("No `indices` => no Cluster. {history:?}"),
            1 => {
                let s = Singleton { center: indices[0] };
                Self {
                    space,
                    history,
                    cardinality: 1,
                    variant: ClusterVariant::Singleton(s),
                    contents: ClusterContents::Indices(indices),
                    ratios: None,
                    t: Default::default(),
                    naive_radius: 0.,
                    scaled_radius: 0.,
                    build_cache: None,
                }
            }
            2 => {
                let [a, b] = [indices[0], indices[1]];
                let diameter = space.one_to_one(a, b);
                let radius = diameter / 2.;
                let d = Dipole {
                    diameter,
                    radius,
                    lfd: 1.,
                    poles: [a, b],
                };
                Self {
                    space,
                    history,
                    cardinality: 2,
                    variant: ClusterVariant::Dipole(d),
                    contents: ClusterContents::Indices(indices),
                    ratios: None,
                    t: Default::default(),
                    naive_radius: diameter,
                    scaled_radius: radius,
                    build_cache: Some([vec![0., diameter], vec![diameter, 0.]]),
                }
            }
            _ => {
                let cardinality = indices.len();

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
                    let m_distances = space.one_to_many(m, &indices);
                    let (a, naive_radius) = helpers::arg_max(&m_distances);
                    (indices[a], naive_radius)
                };
                if naive_radius < EPSILON {
                    let s = Singleton { center: m };
                    return Self {
                        space,
                        history,
                        cardinality,
                        variant: ClusterVariant::Singleton(s),
                        contents: ClusterContents::Indices(indices),
                        ratios: None,
                        t: Default::default(),
                        naive_radius,
                        scaled_radius: naive_radius,
                        build_cache: None,
                    };
                }
                let a_distances = space.one_to_many(a, &indices);

                // the instance farthest from `a`
                let (b, ab) = {
                    let (b, ab) = helpers::arg_max(&a_distances);
                    (indices[b], ab)
                };
                let b_distances = space.one_to_many(b, &indices);
                let radius = ab / 2.;

                // Make triangles to find `c`, the instance which makes maximizes `cm` where `m` is the mid-point of `a` and `b`.
                let mut triangles = indices
                    .iter()
                    .zip(a_distances.iter())
                    .zip(b_distances.iter())
                    .filter(|((&i, &ac), &bc)| {
                        i != a && i != b && ac > EPSILON && ab > EPSILON && makes_triangle([ab, ac, bc])
                    })
                    .map(|((_, &ac), &bc)| Triangle::with_edges_unchecked([ab, ac, bc]))
                    .collect::<Vec<_>>();

                if triangles.is_empty() {
                    // either there are only two unique instances or all instances are colinear
                    let radial_distances = a_distances.iter().map(|&d| (d - radius).abs()).collect::<Vec<_>>();
                    let lfd = helpers::get_lfd(radius, &radial_distances);

                    let d = Dipole {
                        diameter: ab,
                        radius,
                        lfd,
                        poles: [a, b],
                    };
                    return Self {
                        space,
                        history,
                        cardinality,
                        variant: ClusterVariant::Dipole(d),
                        contents: ClusterContents::Indices(indices),
                        ratios: None,
                        t: Default::default(),
                        naive_radius,
                        scaled_radius: radius,
                        build_cache: Some([a_distances, b_distances]),
                    };
                }

                let radial_distances = triangles.iter().map(|abc| abc.cm_sq().sqrt()).collect::<Vec<_>>();
                let lfd = helpers::get_lfd(radius, &radial_distances);

                let (c, cm) = helpers::arg_max(&radial_distances);
                let contents = ClusterContents::Indices(indices);

                let variant = if cm <= radius {
                    ClusterVariant::Dipole(Dipole {
                        diameter: ab,
                        radius,
                        lfd,
                        poles: [a, b],
                    })
                } else {
                    ClusterVariant::Trigon(Trigon {
                        radius,
                        lfd,
                        poles: [a, b],
                        triangle: triangles.swap_remove(c),
                    })
                };

                Self {
                    space,
                    history,
                    cardinality,
                    variant,
                    contents,
                    ratios: None,
                    t: Default::default(),
                    naive_radius,
                    scaled_radius: radius * 3_f64.sqrt(),
                    build_cache: Some([a_distances, b_distances]),
                }
            }
        }
    }

    pub fn partition(mut self, partition_criteria: &PartitionCriteria<'a, T, S>, recursive: bool) -> Self {
        if partition_criteria.check(&self) {
            let [left, right] = self.partition_once();

            let (left, right) = if recursive {
                // (
                //     left.partition(partition_criteria, recursive),
                //     right.partition(partition_criteria, recursive),
                // )
                rayon::join(
                    || left.partition(partition_criteria, recursive),
                    || right.partition(partition_criteria, recursive),
                )
            } else {
                (left, right)
            };
            self.contents = ClusterContents::Children([Box::new(left), Box::new(right)]);
        }
        self.build_cache = None;
        self
    }

    fn partition_once(&self) -> [Self; 2] {
        let indices = match &self.contents {
            ClusterContents::Indices(indices) => indices,
            _ => panic!("Impossible!"),
        };

        let [a, b] = self.variant.poles().unwrap();

        let [a_distances, b_distances] = self.build_cache.as_ref().unwrap();
        let lefties = indices
            .iter()
            .zip(a_distances.iter())
            .zip(b_distances.iter())
            .filter(|((&i, _), _)| i != a && i != b)
            .map(|((_, &l), &r)| l <= r)
            .collect::<Vec<_>>();

        let mut right_indices = indices
            .iter()
            .filter(|&&i| i != a && i != b)
            .zip(lefties.iter())
            .filter(|(_, &b)| !b)
            .map(|(&i, _)| i)
            .collect::<Vec<_>>();
        right_indices.push(b);

        let mut left_indices = indices
            .iter()
            .filter(|&&i| i != a && i != b)
            .zip(lefties.iter())
            .filter(|(_, &b)| b)
            .map(|(&i, _)| i)
            .collect::<Vec<_>>();
        left_indices.push(a);

        let (left_indices, right_indices) = if left_indices.len() < right_indices.len() {
            (right_indices, left_indices)
        } else {
            (left_indices, right_indices)
        };

        let left_history = {
            let mut history = self.history.clone();
            history.push(false);
            history
        };
        let right_history = {
            let mut history = self.history.clone();
            history.push(true);
            history
        };

        let left = Self::new(self.space, left_history, left_indices);
        let right = Self::new(self.space, right_history, right_indices);

        [left, right]
    }

    #[allow(unused_mut, unused_variables)]
    pub fn with_ratios(mut self, normalized: bool) -> Self {
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
            ClusterContents::Indices(indices) => indices.clone(),
            ClusterContents::Children([left, right]) => {
                left.indices().into_iter().chain(right.indices().into_iter()).collect()
            }
        }
    }

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

    pub fn variant_name(&self) -> &str {
        self.variant.name()
    }

    pub fn depth(&self) -> usize {
        self.history.len() - 1
    }

    pub fn is_singleton(&self) -> bool {
        matches!(self.variant, ClusterVariant::Singleton(_))
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self.contents, ClusterContents::Indices(_))
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

    pub fn children(&self) -> Option<[&Self; 2]> {
        match &self.contents {
            ClusterContents::Indices(_) => None,
            ClusterContents::Children([left, right]) => Some([left.as_ref(), right.as_ref()]),
        }
    }

    pub fn ratios(&self) -> Ratios {
        self.ratios
            .expect("Please call `with_ratios` after `build` before using this method.")
    }

    pub fn subtree(&self) -> Vec<&Self> {
        let subtree = vec![self];
        match &self.contents {
            ClusterContents::Indices(_) => subtree,
            ClusterContents::Children([left, right]) => subtree
                .into_iter()
                .chain(left.subtree().into_iter())
                .chain(right.subtree().into_iter())
                .collect(),
        }
    }

    pub fn num_descendants(&self) -> usize {
        self.subtree().len() - 1
    }

    pub fn max_leaf_depth(&self) -> usize {
        self.subtree().into_iter().map(|c| c.depth()).max().unwrap()
    }

    fn query_triangle(&self, ab: f64, query: &[T], [a, b]: [usize; 2]) -> f64 {
        let [aq, bq] = {
            let distances = self.space.query_to_many(query, &[a, b]);
            [distances[0], distances[1]]
        };
        if makes_triangle([ab, aq, bq]) {
            Triangle::with_edges_unchecked([ab, aq, bq]).cm_sq().sqrt()
        } else if aq < bq {
            aq
        } else {
            bq
        }
    }

    pub fn distance_to_query(&self, query: &[T]) -> f64 {
        match &self.variant {
            ClusterVariant::Singleton(s) => self.space.query_to_one(query, s.center),
            ClusterVariant::Dipole(d) => self.query_triangle(d.diameter, query, d.poles),
            ClusterVariant::Trigon(t) => self.query_triangle(t.triangle.edge_lengths()[0], query, t.poles),
        }
    }
}

#[derive(Debug)]
enum ClusterContents<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    Indices(Vec<usize>),
    Children([Box<Cluster<'a, T, S>>; 2]),
}

#[derive(Debug)]
enum ClusterVariant {
    Singleton(Singleton),
    Dipole(Dipole),
    Trigon(Trigon),
}

impl ClusterVariant {
    fn name(&self) -> &str {
        match self {
            ClusterVariant::Singleton(_) => "singleton",
            ClusterVariant::Dipole(_) => "dipole",
            ClusterVariant::Trigon(_) => "trigon",
        }
    }
}

trait Variant {
    fn radius(&self) -> f64;
    fn lfd(&self) -> f64;
    fn poles(&self) -> Option<[usize; 2]> {
        None
    }
    fn center(&self) -> Option<usize> {
        None
    }
    fn triangle(&self) -> Option<&Triangle> {
        None
    }
}

impl Variant for ClusterVariant {
    fn radius(&self) -> f64 {
        match self {
            Self::Singleton(v) => v.radius(),
            Self::Dipole(v) => v.radius(),
            Self::Trigon(v) => v.radius(),
        }
    }

    fn lfd(&self) -> f64 {
        match self {
            Self::Singleton(v) => v.lfd(),
            Self::Dipole(v) => v.lfd(),
            Self::Trigon(v) => v.lfd(),
        }
    }

    fn poles(&self) -> Option<[usize; 2]> {
        match self {
            Self::Singleton(v) => v.poles(),
            Self::Dipole(v) => v.poles(),
            Self::Trigon(v) => v.poles(),
        }
    }

    fn center(&self) -> Option<usize> {
        match self {
            Self::Singleton(v) => v.center(),
            Self::Dipole(v) => v.center(),
            Self::Trigon(v) => v.center(),
        }
    }

    fn triangle(&self) -> Option<&Triangle> {
        match self {
            Self::Singleton(v) => v.triangle(),
            Self::Dipole(v) => v.triangle(),
            Self::Trigon(v) => v.triangle(),
        }
    }
}

#[derive(Debug)]
struct Singleton {
    center: usize,
}

impl Variant for Singleton {
    fn radius(&self) -> f64 {
        0.
    }

    fn lfd(&self) -> f64 {
        1.
    }

    fn center(&self) -> Option<usize> {
        Some(self.center)
    }
}

#[derive(Debug)]
struct Dipole {
    diameter: f64,
    radius: f64,
    lfd: f64,
    poles: [usize; 2],
}

impl Variant for Dipole {
    fn radius(&self) -> f64 {
        self.radius
    }

    fn lfd(&self) -> f64 {
        self.lfd
    }

    fn poles(&self) -> Option<[usize; 2]> {
        Some(self.poles)
    }
}

#[derive(Debug)]
struct Trigon {
    radius: f64,
    lfd: f64,
    poles: [usize; 2],
    triangle: Triangle,
}

impl Variant for Trigon {
    fn radius(&self) -> f64 {
        self.radius
    }

    fn lfd(&self) -> f64 {
        self.lfd
    }

    fn poles(&self) -> Option<[usize; 2]> {
        Some(self.poles)
    }

    fn triangle(&self) -> Option<&Triangle> {
        Some(&self.triangle)
    }
}

//! Provides the `Metric` trait and implementations for some common distance
//! functions.

use num_traits::NumCast;
use rayon::prelude::*;

use crate::prelude::*;

/// A `Metric` is a function that takes two instances (over a `Number` T) from a
/// `Dataset` and deterministically produces a non-negative `Number` U.
pub trait Metric<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    /// Returns the name of the `Metric` as a String.
    fn name(&self) -> &str;

    /// Returns the distance between two instances.
    fn one_to_one(&self, x: &[T], y: &[T]) -> U;

    fn one_to_many(&self, x: &[T], ys: &[&[T]]) -> Vec<U> {
        ys.iter().map(|y| self.one_to_one(x, y)).collect()
    }

    fn par_one_to_many(&self, x: &[T], ys: &[&[T]]) -> Vec<U> {
        ys.par_iter().map(|y| self.one_to_one(x, y)).collect()
    }

    fn many_to_many(&self, xs: &[&[T]], ys: &[&[T]]) -> Vec<Vec<U>> {
        xs.iter().map(|x| self.one_to_many(x, ys)).collect()
    }

    fn par_many_to_many(&self, xs: &[&[T]], ys: &[&[T]]) -> Vec<Vec<U>> {
        xs.par_iter().map(|x| self.one_to_many(x, ys)).collect()
    }

    // TODO: Make this faster by computing only the lower triangle
    fn pairwise(&self, is: &[&[T]]) -> Vec<Vec<U>> {
        self.many_to_many(is, is)
    }

    fn par_pairwise(&self, is: &[&[T]]) -> Vec<Vec<U>> {
        self.par_many_to_many(is, is)
    }

    /// Whether the metric is expensive to compute.
    fn is_cheap(&self) -> bool;
}

pub fn cheap<T: Number, U: Number>(name: &str) -> Result<&dyn Metric<T, U>, String> {
    match name {
        "euclidean" => Ok(&Euclidean { is_cheap: true }),
        "euclideansq" => Ok(&EuclideanSq { is_cheap: true }),
        "manhattan" => Ok(&Manhattan { is_cheap: true }),
        "cosine" => Ok(&Cosine { is_cheap: true }),
        "hamming" => Ok(&Hamming { is_cheap: true }),
        "jaccard" => Ok(&Jaccard { is_cheap: true }),
        _ => Err(format!("{} is not a metric we provide.", name)),
    }
}

pub fn expensive<T: Number, U: Number>(name: &str) -> Result<&dyn Metric<T, U>, String> {
    match name {
        "euclidean" => Ok(&Euclidean { is_cheap: false }),
        "euclideansq" => Ok(&EuclideanSq { is_cheap: false }),
        "manhattan" => Ok(&Manhattan { is_cheap: false }),
        "cosine" => Ok(&Cosine { is_cheap: false }),
        "hamming" => Ok(&Hamming { is_cheap: false }),
        "jaccard" => Ok(&Jaccard { is_cheap: false }),
        _ => Err(format!("{} is not a metric we provide.", name)),
    }
}

/// L2-norm.
#[derive(Debug)]
pub struct Euclidean {
    pub is_cheap: bool,
}

impl<T: Number, U: Number> Metric<T, U> for Euclidean {
    fn name(&self) -> &str {
        "euclidean"
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let d: T = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        let d: f64 = NumCast::from(d).unwrap();
        U::from(d.sqrt()).unwrap()
    }

    fn is_cheap(&self) -> bool {
        self.is_cheap
    }
}

/// Squared L2-norm.
#[derive(Debug)]
pub struct EuclideanSq {
    pub is_cheap: bool,
}

impl<T: Number, U: Number> Metric<T, U> for EuclideanSq {
    fn name(&self) -> &str {
        "euclideansq"
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let d: T = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        U::from(d).unwrap()
    }

    fn is_cheap(&self) -> bool {
        self.is_cheap
    }
}

/// L1-norm.
#[derive(Debug)]
pub struct Manhattan {
    pub is_cheap: bool,
}

impl<T: Number, U: Number> Metric<T, U> for Manhattan {
    fn name(&self) -> &str {
        "manhattan"
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let d: T = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| if a > b { a - b } else { b - a })
            .sum();
        U::from(d).unwrap()
    }

    fn is_cheap(&self) -> bool {
        self.is_cheap
    }
}

/// 1 - cosine-similarity.
#[derive(Debug)]
pub struct Cosine {
    pub is_cheap: bool,
}

impl<T: Number, U: Number> Metric<T, U> for Cosine {
    fn name(&self) -> &str {
        "cosine"
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let (xx, yy, xy) = x.iter().zip(y.iter()).fold(
            (T::zero(), T::zero(), T::zero()),
            |(xx, yy, xy), (&a, &b)| (xx + a * a, yy + b * b, xy + a * b),
        );

        if xx == T::zero() || yy == T::zero() || xy <= T::zero() {
            return U::one();
        }

        U::from(1. - xy.as_f64() / (xx * yy).as_f64().sqrt()).unwrap()
    }

    fn is_cheap(&self) -> bool {
        self.is_cheap
    }
}

/// Count of differences at each indexed feature. This is not normalized by the
/// number of features.
#[derive(Debug)]
pub struct Hamming {
    pub is_cheap: bool,
}

impl<T: Number, U: Number> Metric<T, U> for Hamming {
    fn name(&self) -> &str {
        "hamming"
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let d = x.iter().zip(y.iter()).filter(|(&a, &b)| a != b).count();
        U::from(d).unwrap()
    }

    fn is_cheap(&self) -> bool {
        self.is_cheap
    }
}

/// 1 - jaccard-similarity.
///
/// Warning: DO NOT use this with floating-point numbers.
#[derive(Debug)]
pub struct Jaccard {
    pub is_cheap: bool,
}

impl<T: Number, U: Number> Metric<T, U> for Jaccard {
    fn name(&self) -> &str {
        "jaccard"
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        if x.is_empty() || y.is_empty() {
            return U::one();
        }

        let x = std::collections::HashSet::<u64>::from_iter(
            x.iter().map(|&a| NumCast::from(a).unwrap()),
        );
        let y = std::collections::HashSet::from_iter(y.iter().map(|&a| NumCast::from(a).unwrap()));

        let intersection = x.intersection(&y).count();

        if intersection == x.len() && intersection == y.len() {
            return U::zero();
        }

        let union = x.union(&y).count();

        U::one() - U::from(intersection as f64 / union as f64).unwrap()
    }

    fn is_cheap(&self) -> bool {
        self.is_cheap
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    #[test]
    fn test_on_two() {
        let a = vec![1., 2., 3.];
        let b = vec![3., 3., 1.];

        let metric = super::cheap("euclideansq").unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 9.);

        let metric = super::cheap("euclidean").unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 3.);

        let metric = super::cheap("manhattan").unwrap();
        approx_eq!(f64, metric.one_to_one(&a, &a), 0.);
        approx_eq!(f64, metric.one_to_one(&a, &b), 5.);
    }
}

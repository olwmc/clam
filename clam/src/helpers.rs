//! Utility functions for CLAM and its dependents.

use crate::prelude::*;
use crate::constants;

pub fn arg_min<T: PartialOrd + Copy>(values: &[T]) -> (usize, T) {
    values.iter().enumerate().fold(
        (0, values[0]),
        |(i_min, v_min), (i, &v)| {
            if v < v_min {
                (i, v)
            } else {
                (i_min, v_min)
            }
        },
    )
}

pub fn arg_max<T: PartialOrd + Copy>(values: &[T]) -> (usize, T) {
    values.iter().enumerate().fold(
        (0, values[0]),
        |(i_max, v_max), (i, &v)| {
            if v > v_max {
                (i, v)
            } else {
                (i_max, v_max)
            }
        },
    )
}

pub fn mean<T: Number>(values: &[T]) -> f64 {
    values.iter().cloned().sum::<T>().as_f64() / values.len() as f64
}

pub fn sd<T: Number>(values: &[T], mean: f64) -> f64 {
    values
        .iter()
        .map(|v| v.as_f64())
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        .sqrt()
        / (values.len() as f64)
}

pub fn normalize_1d(values: &[f64]) -> Vec<f64> {
    let mean = mean(values);
    let std = constants::EPSILON + sd(values, mean);
    values
        .iter()
        .map(|value| (value - mean) / (std * 2_f64.sqrt()))
        .map(libm::erf)
        .map(|value| (1. + value) / 2.)
        .collect()
}

pub fn compute_lfd<T: Number>(distances: &[T], radius: T) -> f64 {
    if radius == T::zero() {
        1.
    } else {
        let half_count = distances
            .iter()
            .filter(|&&d| d <= (radius / T::from(2).unwrap()))
            .count();
        if half_count > 0 {
            ((distances.len() as f64) / (half_count as f64)).log2()
        } else {
            1.
        }
    }
}

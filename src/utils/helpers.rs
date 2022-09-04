use crate::prelude::*;

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
    let std = 1e-8 + sd(values, mean);
    values
        .iter()
        .map(|value| (value - mean) / (std * 2_f64.sqrt()))
        .map(libm::erf)
        .map(|value| (1. + value) / 2.)
        .collect()
}

pub fn compute_lfd<U: Number>(points: &[U]) -> f64 {
    let mut points = points.iter().map(|d| d.as_f64()).map(|d| if d < 0. { 0. } else { d }).collect::<Vec<_>>();
    points.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let &r = points.last().unwrap();
    if r == 0. {
        1.
    } else {
        let n = points.len() as f64;
        let lfds = points
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d < r)
            .map(|(i, &d)| {
                if d == 0. || d == r {
                    1.
                } else {
                    let lfd = ((i + 1) as f64 / n).log2() / (d / r).log2();
                    assert!(lfd.is_finite(), "points were {:?}, i was {}, n was {}, d was {} and r was {}", points, i, n, d, r);
                    lfd
                }
            })
            .inspect(|&v| assert!(v.is_finite(), "points were {:?}, n was {} and r was {}", points, n, r))
            .collect::<Vec<_>>();
        mean(&lfds)
    }
}

// pub fn normalize_2d(values: Array2<f64>, on_rows: bool) -> Array2<f64> {
//     let shape = (values.nrows(), values.ncols());
//     let axis = Axis(if on_rows { 0 } else { 1 });
//     let values: Vec<_> = values
//         .axis_iter(axis)
//         .into_par_iter()
//         .flat_map(|values| normalize_1d(&values.to_vec()))
//         .collect(); // this is now in col-major order.
//     let values = (0..shape.0)
//         .map(|r| values.iter().skip(r))
//         .flat_map(|row| row.step_by(shape.0))
//         .cloned()
//         .collect();
//     Array2::from_shape_vec(shape, values).unwrap()
// }

pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use clam::prelude::*;

use utils::search_readers;

fn partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise");
    group
        .significance_level(0.05)
        .measurement_time(std::time::Duration::new(10, 0));

    let (data_name, metric_name) = ("fashion-mnist", "euclidean");
    let (features, _) = search_readers::read_search_data(data_name).unwrap();

    let dataset = clam::Tabular::new(&features, data_name.to_string());

    let metric = metric_from_name::<f32, f32>(metric_name, false).unwrap();
    let space = clam::TabularSpace::new(&dataset, metric.as_ref(), false);

    let bench_name = format!("{}-{}-{}", data_name, dataset.cardinality(), dataset.dimensionality());

    for n in [10, 100, 1_000] {
        group.bench_with_input(format!("{}-pairwise-{}", bench_name, n), &n, |b, &n| {
            let indices = (0..n).collect::<Vec<_>>();
            b.iter_with_large_drop(|| space.pairwise(&indices));
        });

        group.bench_with_input(format!("{}-many_to_many-{}", bench_name, n), &n, |b, &n| {
            let indices = (0..n).collect::<Vec<_>>();
            b.iter_with_large_drop(|| space.many_to_many(&indices, &indices));
        });
    }

    group.finish();
}

criterion_group!(benches, partition);
criterion_main!(benches);

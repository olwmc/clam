pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use clam::prelude::*;

use utils::search_readers;

fn cakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn-search");
    group
        .significance_level(0.05)
        .measurement_time(std::time::Duration::new(10, 0)) // 10 seconds
        .sample_size(30);

    for &(data_name, metric_name) in search_readers::SEARCH_DATASETS.iter() {
        if !data_name.contains("mnist") {
            continue;
        }

        let (train, test) = search_readers::read_search_data(data_name).unwrap();

        let train = clam::Tabular::new(&train, data_name.to_string());
        let test = clam::Tabular::new(&test, data_name.to_string());

        let queries = (0..100)
            .map(|i| test.get(i % test.cardinality()))
            .collect::<Vec<_>>();

        let metric = metric_from_name::<f32, f32>(metric_name, false).unwrap();
        let space = clam::TabularSpace::new(&train, metric.as_ref(), false);
        let partition_criteria = clam::PartitionCriteria::default();
        let cakes = clam::CAKES::new(&space).build(&partition_criteria);

        let radius = cakes.radius();
        let radii_factors = (5..25).step_by(5).chain((25..=100).step_by(25)).collect::<Vec<_>>();

        let bench_name = format!(
            "{}-{}-{}-{}",
            data_name,
            train.cardinality(),
            train.dimensionality(),
            metric_name
        );
        for factor in radii_factors {
            group.bench_with_input(BenchmarkId::new(&bench_name, factor), &factor, |b, &factor| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius / (factor as f32)))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);

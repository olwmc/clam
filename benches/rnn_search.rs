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
        if data_name != "fashion-mnist" {
            continue;
        }

        let (train, test) = search_readers::read_search_data(data_name).unwrap();

        let train = clam::Tabular::new(&train, data_name.to_string());
        let test = clam::Tabular::new(&test, data_name.to_string());

        let queries = (0..test.cardinality()).map(|i| test.get(i)).collect::<Vec<_>>();

        let metric = metric_from_name::<f32, f32>(metric_name, false).unwrap();
        let space = clam::TabularSpace::new(&train, metric.as_ref(), false);
        let partition_criteria = clam::PartitionCriteria::default();
        let cakes = clam::CAKES::new(&space).build(&partition_criteria);

        // let radius = cakes.radius();
        // let radii_factors = (10..50)
        //     .step_by(10)
        //     .chain((50..250).step_by(50))
        //     .chain((250..=1000).step_by(250))
        //     .collect::<Vec<_>>();
        let radii = [10., 100., 250., 500., 1000.];

        let bench_name = format!(
            "{}-{}-{}-{}-{}",
            data_name,
            train.cardinality(),
            train.dimensionality(),
            metric_name,
            queries.len(),
        );

        for radius in radii {
            group.bench_with_input(BenchmarkId::new(&bench_name, radius), &radius, |b, &radius| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);

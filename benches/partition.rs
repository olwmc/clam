pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use clam::prelude::*;

use utils::anomaly_readers;

fn partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition");
    group
        .significance_level(0.05)
        .measurement_time(std::time::Duration::new(10, 0));

    for &data_name in anomaly_readers::ANOMALY_DATASETS.iter() {
        let (features, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();

        let dataset = clam::Tabular::new(&features, data_name.to_string());

        let metric = metric_from_name::<f32, f32>("euclidean", false).unwrap();
        let partition_criteria = clam::PartitionCriteria::default();
        let space = clam::TabularSpace::new(&dataset, metric.as_ref(), false);

        let bench_name = format!("{}-{}-{}", data_name, dataset.cardinality(), dataset.dimensionality());
        group.bench_function(&bench_name, |b| {
            b.iter_with_large_drop(|| Cluster::new_root(&space).build().partition(&partition_criteria, true))
        });
    }

    group.finish();
}

criterion_group!(benches, partition);
criterion_main!(benches);

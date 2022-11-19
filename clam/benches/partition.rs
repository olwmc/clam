pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use utils::readers;

fn partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Partition");
    group
        .significance_level(0.05)
        // .measurement_time(std::time::Duration::new(60, 0));
        .sample_size(10);

    for &(data_name, metric_name) in readers::DATASETS.iter() {
        if data_name == "kosarak" || data_name == "lastfm" {
            continue;
        }

        let (train, _) = readers::read_data(data_name).unwrap();
        if train.len() > 100_000 {
            continue;
        }

        let train_data = clam::Tabular::new(&train, data_name.to_string());

        let metric: &dyn clam::Metric<f32, f32> = if metric_name == "euclidean" {
            &clam::Euclidean { is_expensive: false }
        } else if metric_name == "cosine" {
            &clam::Cosine { is_expensive: false }
        } else if metric_name == "jaccard" {
            &clam::Jaccard { is_expensive: false }
        } else {
            panic!()
        };

        // let log_cardinality = (train.cardinality() as f64).log2() as usize;
        let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1);
        let space = clam::TabularSpace::new(&train_data, metric, false);

        group.bench_function(data_name, |b| {
            b.iter_with_large_drop(|| clam::Cluster::new_root(&space).build().partition(&partition_criteria, true))
        });
    }

    group.finish();
}

criterion_group!(benches, partition);
criterion_main!(benches);

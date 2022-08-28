use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use clam::prelude::*;

pub mod utils;
use utils::search_readers;

fn cakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn-search");
    group
        .significance_level(0.05)
        .measurement_time(std::time::Duration::new(10, 0)) // 10 seconds
        .sample_size(30);

    for &(data_name, metric_name) in search_readers::SEARCH_DATASETS.iter() {
        // if metric_name != "euclidean" {
        //     continue;
        // }
        if !data_name.contains("mnist") {
            continue;
        }

        let (train, test) = search_readers::read_search_data(data_name).unwrap();

        let train = clam::Tabular::new(&train, data_name.to_string());
        let test = clam::Tabular::new(&test, data_name.to_string());

        let queries = (0..100).map(|i| test.get(i % test.cardinality())).collect::<Vec<_>>();

        let metric = metric_from_name::<f32, f32>(metric_name, false).unwrap();
        let space = clam::TabularSpace::new(&train, metric.as_ref(), false);
        let partition_criteria = clam::PartitionCriteria::default();
        let cakes = clam::CAKES::new(&space).build(&partition_criteria);

        let ks = [1, 10, 100];

        let bench_name = format!(
            "{}-{}-{}-{}",
            data_name,
            train.cardinality(),
            train.dimensionality(),
            metric_name
        );
        for k in ks {
            if k > train.cardinality() {
                continue;
            }

            // queries.iter().for_each(|&query| {
            //     let indices = cakes.knn_search(query, k);
            //     assert!(indices.len() >= k, "{} vs {}", indices.len(), k);
            //     if indices.len() > k {
            //         let mut distances = space.query_to_many(query, &indices);
            //         distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            //         let kth = distances[k - 1];
            //         assert!(
            //             distances[k..].iter().all(|d| approx_eq!(f32, *d, kth)),
            //             "{:?}",
            //             &distances
            //         );
            //     }
            // });

            group.bench_with_input(BenchmarkId::new(&bench_name, k), &k, |b, &k| {
                b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use crate::{
        cluster::{Cluster, PartitionCriteria},
        dataset::arrow_dataset::io::generate_batched_arrow_test_data,
        dataset::{BatchedArrowDataset, Dataset},
    };

    #[test]
    fn grab_col_raw() {
        let batches = 3;
        let cols_per_batch = 2;
        let dimensionality = 4;
        let seed = 25565;

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed));

        let name = "Test Dataset".to_string();
        let dataset =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, crate::distances::f32::euclidean, false).unwrap();

        assert_eq!(dataset.cardinality(), batches * cols_per_batch);
    }

    #[test]
    fn test_cluster() {
        let batches = 1;
        let cols_per_batch = 4;
        let dimensionality = 3;
        let seed = 25565;

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed));

        let name = "Test Dataset".to_string();
        let data =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, crate::distances::f32::euclidean, false).unwrap();

        let indices = data.indices().to_vec();
        let partition_criteria = PartitionCriteria::new(true).with_max_depth(3).with_min_cardinality(1);
        let cluster = Cluster::new_root(&data, &indices, Some(42)).partition(&data, &partition_criteria, true);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 4);
        assert_eq!(cluster.num_descendants(), 6);
        assert!(cluster.radius() > 0.);
        assert_eq!(format!("{cluster}"), "1");

        let [left, right] = cluster.children().unwrap();
        assert_eq!(format!("{left}"), "2");
        assert_eq!(format!("{right}"), "3");
    }

    // Tests the difference between our implementation and the arrow2 implementation
    #[test]
    fn test_diff() {
        use float_cmp::approx_eq;

        let dimensionality = 50;
        let cols_per_batch = 500;

        let path = generate_batched_arrow_test_data(1, dimensionality, cols_per_batch, Some(42));
        let mut reader = std::fs::File::open(path.join("batch-0.arrow")).unwrap();
        let metadata = arrow2::io::ipc::read::read_file_metadata(&mut reader).unwrap();
        let mut reader = arrow2::io::ipc::read::FileReader::new(reader, metadata, None, None);

        let binding = reader.next().unwrap().unwrap();
        let columns = binding.columns();

        let name = "Test Dataset".to_string();
        let data =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, crate::distances::f32::euclidean, false).unwrap();

        for i in 0..cols_per_batch {
            let col: Vec<f32> = columns[i]
                .as_any()
                .downcast_ref::<arrow2::array::PrimitiveArray<f32>>()
                .unwrap()
                .iter()
                .map(|x| *x.unwrap())
                .collect();

            for j in 0..dimensionality {
                approx_eq!(f32, col[j], data.get(i)[j]);
            }
        }
    }

    #[test]
    fn test_reorder() {
        let dimensionality = 1;
        let cols_per_batch = 10;

        let path = generate_batched_arrow_test_data(1, dimensionality, cols_per_batch, Some(42));
        let name = "Test Dataset".to_string();
        let mut data =
            BatchedArrowDataset::new(path.to_str().unwrap(), name, crate::distances::f32::euclidean, false).unwrap();

        let reordering = [9, 8, 7, 6, 5, 4, 3, 2, 1];
        data.reorder(&reordering);
        assert_eq!(data.reordered_indices(), reordering);
    }
}

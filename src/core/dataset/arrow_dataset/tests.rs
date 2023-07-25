#[cfg(test)]
mod tests {
    use crate::{
        cluster::{Cluster, PartitionCriteria},
        dataset::arrow_dataset::io::generate_batched_arrow_test_data,
        dataset::{BatchedArrowReader, Dataset},
    };

    #[test]
    fn grab_col_raw() {
        // Construct the batched reader
        let batches = 3;
        let cols_per_batch = 10;
        let dimensionality = 3;
        let seed = 25565;

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed));

        let dataset = BatchedArrowReader::new(path.to_str().unwrap(), crate::distances::f32::euclidean).unwrap();

        assert_eq!(dataset.cardinality(), batches * cols_per_batch);
    }

    #[test]
    fn test_cluster() {
        let batches = 1;
        let cols_per_batch = 4;
        let dimensionality = 3;
        let seed = 25565;

        let path = generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed));

        let data = BatchedArrowReader::new(path.to_str().unwrap(), crate::distances::f32::euclidean).unwrap();

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
}

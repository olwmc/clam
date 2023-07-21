#[cfg(test)]
mod tests {
    use crate::{
        cluster::{Cluster, PartitionCriteria},
        dataset::{BatchedArrowReader, Dataset},
    };
    use std::{
        fs::{create_dir, remove_dir_all, File},
        path::{Path, PathBuf},
    };

    use arrow2::{
        array::Float32Array,
        //array::
        chunk::Chunk,
        datatypes::{DataType::Float32, Field, Schema},
        io::ipc::write::{FileWriter, WriteOptions},
    };

    use rand::{Rng, SeedableRng};
    use uuid::Uuid;

    /// If seed is given as none, the columns will be generated as follows:
    ///     the first row of each column will be the index of that column
    ///     the following rows will be, for the nth row, the column index + n
    ///
    ///     [ 0 1 2 3 ]
    ///     [ 1 2 3 4 ]
    ///     [ 2 3 4 5 ]
    ///
    /// Returns the path of the newly created dataset
    fn generate_batched_arrow_test_data(
        batches: usize,
        dimensionality: usize,
        cols_per_batch: usize,
        seed: Option<u64>,
    ) -> PathBuf {
        // Open up the system's temp dir
        // We need to create a uuid'd directory like this to allow for rust to run these tests
        // in multiple threads.
        let path = std::env::temp_dir().join(format!("arrow-test-data-{}", Uuid::new_v4().to_string()));

        if Path::exists(&path) {
            let _ = remove_dir_all(path.clone());
        }

        create_dir(path.clone()).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap());

        let fields = (0..dimensionality)
            .map(|x| Field::new(x.to_string(), Float32, false))
            .collect::<Vec<Field>>();

        let schema = Schema::from(fields);

        for batch_number in 0..batches {
            let file = File::create(path.join(format!("batch-{}.arrow", batch_number))).unwrap();
            let options = WriteOptions { compression: None };
            let mut writer = FileWriter::try_new(file, schema.clone(), None, options).unwrap();

            let arrays = (0..cols_per_batch)
                .map(|_| {
                    Float32Array::from_vec((0..dimensionality).map(|_| rng.gen_range(0.0..100_000.0)).collect()).boxed()
                })
                .collect();

            // println!("{:?}", arrays);

            let chunk = Chunk::try_new(arrays).unwrap();
            writer.write(&chunk, None).unwrap();
            writer.finish().unwrap();
        }

        path
    }

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

#[cfg(test)]
mod tests {
    use crate::dataset::{BatchedArrowReader, Dataset};
    use std::{
        fs::{create_dir, remove_dir_all, File},
        path::Path,
    };

    use arrow2::{
        array::UInt32Array,
        chunk::Chunk,
        datatypes::{DataType::UInt32, Field, Schema},
        io::ipc::write::{FileWriter, WriteOptions},
    };
    use rand::{Rng, SeedableRng};

    /// If seed is given as none, the columns will be generated as follows:
    ///     the first row of each column will be the index of that column
    ///     the following rows will be, for the nth row, the column index + n
    /// 
    ///     [ 0 1 2 3 ]
    ///     [ 1 2 3 4 ]
    ///     [ 2 3 4 5 ]
    fn generate_batched_arrow_test_data(batches: usize, dimensionality: usize, cols_per_batch: usize, seed: Option<u64>) {
        // Open up the system's temp dir
        let path = std::env::temp_dir().join("arrow-test-data");
        if Path::exists(&path) {
            remove_dir_all(path.clone()).unwrap();
        }

        create_dir(path.clone()).unwrap();

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap());

        let fields = (0..dimensionality)
            .map(|x| Field::new(x.to_string(), UInt32, false))
            .collect::<Vec<Field>>();

        let schema = Schema::from(fields);

        for batch_number in 0..batches {
            let file = File::create(path.join(format!("batch-{}.arrow", batch_number))).unwrap();
            let options = WriteOptions { compression: None };
            let mut writer = FileWriter::try_new(file, schema.clone(), None, options).unwrap();

            // TODO: Make this randomly generated w seed
            let arrays = (0..cols_per_batch)
                .map(|_| UInt32Array::from_vec((0..dimensionality).map(|_| rng.gen_range(0..100_000)).collect()).boxed())
                .collect();

            let chunk = Chunk::try_new(arrays).unwrap();
            writer.write(&chunk, None).unwrap();
            writer.finish().unwrap();
        }
    }

    #[test]
    fn grab_col_raw() {
        // Construct the batched reader
        let batches = 5;
        let cols_per_batch = 3;
        let dimensionality = 2;
        let seed = 25565;

        generate_batched_arrow_test_data(batches, dimensionality, cols_per_batch, Some(seed));
        let dataset = BatchedArrowReader::new("/tmp/arrow-test-data/", crate::distances::u32::euclidean).unwrap();
        assert_eq!(dataset.cardinality(), batches * cols_per_batch);

        for i in 0..(batches * cols_per_batch) {
            println!("{:?}", dataset.get(i));
        }
    }
}
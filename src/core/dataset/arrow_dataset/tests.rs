#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};
    use arrow2::{io::ipc::read::{read_file_metadata, FileReader}, array::PrimitiveArray};
    use float_cmp::approx_eq;
    use crate::dataset::{Dataset, BatchedArrowReader};
    const DATA_DIR: &str = "/home/olwmc/current/data";
    const METRIC: fn(&[u8], &[u8]) -> f32 = crate::distances::u8::euclidean;

    #[test]
    fn grab_col_raw() {
        // Construct the batched reader
        let dataset = BatchedArrowReader::new(DATA_DIR, METRIC);
        assert_eq!(dataset.cardinality(), 20_000_000);

        for i in 0..10 {
            let column: Vec<u8> = dataset.get(10_000_000+i);
            
            assert_eq!(column.len(), 128);
        }
    }

    #[test]
    fn test_reordering_map() {
        // Construct the batched reader
        let dataset = BatchedArrowReader::new(DATA_DIR, METRIC);
        dataset.write_reordering_map().unwrap();

        drop(dataset);

        let dataset = BatchedArrowReader::new(DATA_DIR, METRIC);

        assert_eq!(dataset.indices().len(), 20_000_000);
        assert_eq!(
            dataset.indices.reordered_indices[0..10],
            (0..10).collect::<Vec<usize>>()
        );
    }

    #[test]
    fn test_space() {
        let dataset = BatchedArrowReader::new(DATA_DIR, METRIC);

        dbg!(dataset.one_to_one(0, 0));
        dbg!(dataset.one_to_one(0, 1));
        dbg!(dataset.one_to_one(1, 0));
        dbg!(dataset.one_to_one(1, 1));
        
        approx_eq!(f32, dataset.one_to_one(0, 0), 0.);
        approx_eq!(f32, dataset.one_to_one(0, 1), 3.);
        approx_eq!(f32, dataset.one_to_one(1, 0), 3.);
        approx_eq!(f32, dataset.one_to_one(1, 1), 0.);
    }

    #[test]
    #[ignore]
    fn grab_col_arrow2() {
        let mut reader = File::open(PathBuf::from(DATA_DIR).join("base-0.arrow")).unwrap();
        let metadata = read_file_metadata(&mut reader).unwrap();
        let mut reader = FileReader::new(reader, metadata, None, None);

        println!("{:?}", reader.next().unwrap().unwrap().columns()[0]);
    }

    #[test]
    #[ignore]
    fn assert_my_code_isnt_useless() {
        // Arrow2
        let arrow_column: Vec<u8> = {
            let mut reader = File::open(PathBuf::from(DATA_DIR).join("base-1.arrow")).unwrap();
            let metadata = read_file_metadata(&mut reader).unwrap();
            let mut reader = FileReader::new(reader, metadata, None, None);

            // There's only one column, so we grab it
            let binding = reader.next().unwrap().unwrap();
            let col = &binding.columns()[0];

            // Convert the arrow column to vec<u8>
            col
                .as_any()
                .downcast_ref::<PrimitiveArray<u8>>()
                .unwrap()
                .iter()
                .map(|x| *x.unwrap() )
                .collect()
        };

        // Raw reading
        let dataset = BatchedArrowReader::new(DATA_DIR, METRIC);
        let raw_column = dataset.get(10_000_000);

        // Now assert that they're actually equal
        assert_eq!(raw_column, arrow_column);
    }
}

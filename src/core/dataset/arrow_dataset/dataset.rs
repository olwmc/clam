/*
(Oliver)
    My apologies to anyone forced to read this code in its current state.

    Per najib: The silent failure on wrong type is fine

    Turn to Arc<RwLock<T>> to deal with mutability problems
       https://doc.rust-lang.org/std/sync/struct.RwLock.html
*/

use super::{
    io::read_bytes_from_file,
    metadata::{extract_metadata, ArrowMetaData},
};
use crate::number::Number;
use arrow_format::ipc::Buffer;
use std::fs::File;
use std::marker::PhantomData;
use std::path::PathBuf;

#[derive(Debug)]
struct ArrowIndices {
    original_indices: Vec<usize>,
    reordered_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct BatchedArrowDataset<T: Number, U: Number> {
    // The directory where the data is stored
    data_dir: PathBuf,

    metadata: ArrowMetaData,
    readers: Vec<File>,
    indices: ArrowIndices,

    #[allow(dead_code)]
    metric: fn(&[T], &[T]) -> U,

    // We allocate a column of the specific number of bytes
    // necessary (type_size * num_rows) at construction to
    // lessen the number of vector allocations we need to do.
    // This might be able to be removed. Unclear.
    _col: Vec<u8>,

    // We'd like to associate this handle with a type, hence the phantomdata
    _t: PhantomData<T>,
}

impl<T: Number, U: Number> BatchedArrowDataset<T, U> {
    pub fn new(data_dir: &str, metric: fn(&[T], &[T]) -> U) -> Self {
        // TODO: This has to be ordered somehow
        let (mut handles, reordered_indices) = super::io::process_directory(&PathBuf::from(data_dir));

        // Load in the necessary metadata from the file
        let metadata = extract_metadata::<T>(&mut handles[0]);

        let original_indices: Vec<usize> = (0..metadata.cardinality * handles.len()).collect();
        let reordered_indices = match reordered_indices {
            Some(indices) => indices,
            None => original_indices.clone(),
        };

        BatchedArrowDataset {
            data_dir: PathBuf::from(data_dir),

            indices: ArrowIndices {
                reordered_indices,
                original_indices,
            },

            metric,
            readers: handles,
            _t: Default::default(),
            _col: vec![0u8; metadata.row_size_in_bytes()],
            metadata,
        }
    }

    // TODO: Wrap this in a Result
    pub fn get(&mut self, index: usize) -> Vec<T> {
        self.get_column(index)
    }

    fn get_column(&mut self, index: usize) -> Vec<T> {
        // Returns the index of the reader associated with the index
        let reader_index: usize = (index - (index % self.metadata.cardinality)) / self.metadata.cardinality;

        // Gets the index relative to a given reader
        let index: usize = index % self.metadata.cardinality;

        // Becuase we're limited to primitive types, we only have to deal with buffer 0 and
        // buffer 1 which are the validity and data buffers respectively. Therefore for every
        // index, there are two buffers associated with that column, the second of which is
        // the data buffer, hence the 2*i+1.
        let data_buffer: Buffer = self.metadata.buffers[index * 2 + 1];

        let offset = self.metadata.start_of_message + data_buffer.offset as u64;

        read_bytes_from_file(&mut self.readers[reader_index], offset, &mut self._col)
    }

    pub fn write_reordering_map(&self) -> Result<(), arrow2::error::Error> {
        let reordered_indices: Vec<u64> = self.indices.reordered_indices.iter().map(|x| *x as u64).collect();

        super::io::write_reordering_map(reordered_indices, &self.data_dir)
    }
}

impl<T: Number, U: Number> crate::dataset::Dataset<T, U> for BatchedArrowDataset<T, U> {
    fn name(&self) -> String {
        format!("Batched Arrow Dataset : {}", self.data_dir.to_str().unwrap())
    }

    fn cardinality(&self) -> usize {
        self.indices.original_indices.len()
    }

    fn dimensionality(&self) -> usize {
        self.metadata.num_rows
    }

    fn is_metric_expensive(&self) -> bool {
        // TODO: Obviously parametrize this
        false
    }

    fn indices(&self) -> &[usize] {
        &self.indices.original_indices
    }

    fn one_to_one(&self, _left: usize, _right: usize) -> U {
        todo!()
    }

    fn query_to_one(&self, _query: &[T], _index: usize) -> U {
        todo!()
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.indices.reordered_indices.swap(i, j);
    }

    fn set_reordered_indices(&mut self, _indices: &[usize]) {
        todo!()
    }

    fn get_reordered_index(&self, _i: usize) -> usize {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use arrow2::{io::ipc::read::{read_file_metadata, FileReader}, array::PrimitiveArray};

    use super::*;
    use crate::dataset::Dataset;
    const DATA_DIR: &str = "/home/olwmc/current/data";
    const METRIC: fn(&[u8], &[u8]) -> f32 = crate::distances::u8::euclidean;

    #[test]
    fn grab_col_raw() {
        // Construct the batched reader
        let mut dataset = BatchedArrowDataset::new(DATA_DIR, METRIC);

        let column: Vec<u8> = dataset.get(10_000_000);
        
        assert_eq!(column.len(), 128);
        assert_eq!(dataset.cardinality(), 20_000_000);
    }

    #[test]
    fn test_reordering_map() {
        // Construct the batched reader
        let dataset = BatchedArrowDataset::new(DATA_DIR, METRIC);
        dataset.write_reordering_map().unwrap();

        drop(dataset);

        let dataset = BatchedArrowDataset::new(DATA_DIR, METRIC);

        assert_eq!(dataset.indices().len(), 20_000_000);
        assert_eq!(
            dataset.indices.reordered_indices[0..10],
            (0..10).collect::<Vec<usize>>()
        );
    }

    #[test]
    #[ignore]
    fn grab_col_arrow2() {
        let mut reader = File::open("/home/olwmc/current/data/base-0.arrow").unwrap();
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
        let mut dataset = BatchedArrowDataset::new(DATA_DIR, METRIC);
        let raw_column = dataset.get(10_000_000);

        // Now assert that they're actually equal
        assert_eq!(raw_column, arrow_column);
    }
}
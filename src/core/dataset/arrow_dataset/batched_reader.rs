/*
(Oliver)
    My apologies to anyone forced to read this code in its current state.

    Per najib: The silent failure on wrong type is fine1
*/
use super::{
    io::{process_data_directory, read_bytes_from_file},
    metadata::ArrowMetaData,
};
use crate::number::Number;
use arrow_format::ipc::Buffer;
use std::path::PathBuf;
use std::{error::Error, marker::PhantomData};
use std::{fs::File, sync::RwLock};

#[derive(Debug)]
struct ArrowIndices {
    original_indices: Vec<usize>,
    reordered_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct BatchedArrowReader<T: Number, U: Number> {
    // The directory where the data is stored
    data_dir: PathBuf,
    name: String,
    metadata: Vec<ArrowMetaData<T>>,
    readers: RwLock<Vec<File>>,
    indices: ArrowIndices,
    metric: fn(&[T], &[T]) -> U,
    metric_is_expensive: bool,

    // We allocate a column of the specific number of bytes
    // necessary (type_size * num_rows) at construction to
    // lessen the number of vector allocations we need to do.
    // This might be able to be removed. Unclear.
    _col: RwLock<Vec<u8>>,

    // We'd like to associate this handle with a type, hence the phantomdata
    _t: PhantomData<T>,
}

impl<T: Number, U: Number> BatchedArrowReader<T, U> {
    // TODO: Implement a "safe" constructor that actually goes through each metadata and doesn't just guess lol
    // We can read the metadata of many files fairly quickly if we assume static type size

    pub fn new(data_dir: &str, metric: fn(&[T], &[T]) -> U) -> Result<Self, Box<dyn Error>> {
        let path = PathBuf::from(data_dir);
        let (mut handles, reordered_indices) = process_data_directory(&path)?;

        // Load in the necessary metadata from the file
        let metadata = ArrowMetaData::<T>::try_from(&mut handles[0])?;

        // Index information
        let original_indices: Vec<usize> = (0..metadata.cardinality * handles.len()).collect();
        let reordered_indices = match reordered_indices {
            Some(indices) => indices,
            None => original_indices.clone(),
        };

        Ok(BatchedArrowReader {
            data_dir: path,
            name: String::from("Dataset"),
            indices: ArrowIndices {
                reordered_indices,
                original_indices,
            },

            metric,
            metric_is_expensive: false,
            readers: RwLock::new(handles),
            _t: Default::default(),
            _col: RwLock::new(vec![0u8; metadata.row_size_in_bytes()]),
            metadata: vec![metadata],
        })
    }

    pub fn get(&self, index: usize) -> Vec<T> {
        let resolved_index = self.indices.reordered_indices[index];
        self.get_column(resolved_index)
    }

    fn get_column(&self, index: usize) -> Vec<T> {
        let metadata = &self.metadata[0];

        // Returns the index of the reader associated with the index
        let reader_index: usize = (index - (index % metadata.cardinality)) / metadata.cardinality;

        // Gets the index relative to a given reader
        let index: usize = index % metadata.cardinality;

        // Becuase we're limited to primitive types, we only have to deal with buffer 0 and
        // buffer 1 which are the validity and data buffers respectively. Therefore for every
        // index, there are two buffers associated with that column, the second of which is
        // the data buffer, hence the 2*i+1.
        let data_buffer: Buffer = metadata.buffers[index * 2 + 1];

        let offset = metadata.start_of_message + data_buffer.offset as u64;

        // We `expect` here because any other result is a total failure
        let mut readers = self.readers.write().expect("Could not access column. Invalid index");
        let mut _col = self
            ._col
            .write()
            .expect("Could not access column buffer. Memory error.");

        read_bytes_from_file(&mut readers[reader_index], offset, &mut _col)
    }

    pub fn write_reordering_map(&self) -> Result<(), Box<dyn Error>> {
        super::io::write_reordering_map(&self.indices.reordered_indices, &self.data_dir)
    }
}

impl<T: Number, U: Number> crate::dataset::Dataset<T, U> for BatchedArrowReader<T, U> {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn cardinality(&self) -> usize {
        self.indices.original_indices.len()
    }

    fn dimensionality(&self) -> usize {
        // We assume dimensionality is constant throughout the dataset
        self.metadata[0].num_rows
    }

    fn is_metric_expensive(&self) -> bool {
        self.metric_is_expensive
    }

    fn indices(&self) -> &[usize] {
        &self.indices.original_indices
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        (self.metric)(&self.get(left), &self.get(right))
    }

    fn query_to_one(&self, query: &[T], index: usize) -> U {
        (self.metric)(query, &self.get(index))
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.indices.reordered_indices.swap(i, j);
    }

    fn set_reordered_indices(&mut self, indices: &[usize]) {
        self.indices.reordered_indices = indices.to_vec();
    }

    fn get_reordered_index(&self, i: usize) -> usize {
        self.indices.reordered_indices[i]
    }
}

/*
(Oliver)
    My apologies to anyone forced to read this code in its current state.

    Per najib: The silent failure on wrong type is fine

*/

use super::{
    io::{read_bytes_from_file, process_data_directory},
    metadata::{extract_metadata, ArrowMetaData},
};
use crate::number::Number;
use arrow_format::ipc::Buffer;
use std::{fs::File, sync::RwLock};
use std::marker::PhantomData;
use std::path::PathBuf;

#[derive(Debug)]
pub(crate)struct ArrowIndices {
    pub(crate) original_indices: Vec<usize>,
    pub(crate) reordered_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct BatchedArrowReader<T: Number, U: Number> {
    // The directory where the data is stored
    pub(crate) data_dir: PathBuf,

    pub(crate) metadata: ArrowMetaData,
    pub(crate) readers: RwLock<Vec<File>>,
    pub(crate) indices: ArrowIndices,
    pub(crate) metric: fn(&[T], &[T]) -> U,

    // We allocate a column of the specific number of bytes
    // necessary (type_size * num_rows) at construction to
    // lessen the number of vector allocations we need to do.
    // This might be able to be removed. Unclear.
    pub(crate) _col: RwLock<Vec<u8>>,

    // We'd like to associate this handle with a type, hence the phantomdata
    pub(crate) _t: PhantomData<T>,
}

impl<T: Number, U: Number> BatchedArrowReader<T, U> {
    pub fn new(data_dir: &str, metric: fn(&[T], &[T]) -> U) -> Self {
        let (mut handles, reordered_indices) = process_data_directory(&PathBuf::from(data_dir));

        // Load in the necessary metadata from the file
        let metadata = extract_metadata::<T>(&mut handles[0]);

        let original_indices: Vec<usize> = (0..metadata.cardinality * handles.len()).collect();
        let reordered_indices = match reordered_indices {
            Some(indices) => indices,
            None => original_indices.clone(),
        };

        BatchedArrowReader {
            data_dir: PathBuf::from(data_dir),

            indices: ArrowIndices {
                reordered_indices,
                original_indices,
            },

            metric,
            readers: RwLock::new(handles),
            _t: Default::default(),
            _col: RwLock::new(vec![0u8; metadata.row_size_in_bytes()]),
            metadata,
        }
    }

    pub fn get(&self, index: usize) -> Vec<T> {
        let resolved_index = self.indices.reordered_indices[index];
        self.get_column(resolved_index)
    }

    fn get_column(&self, index: usize) -> Vec<T> {
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

        let mut readers = self.readers.write().unwrap();
        let mut _col = self._col.write().unwrap();

        read_bytes_from_file(&mut readers[reader_index], offset, &mut _col)
    }

    pub fn write_reordering_map(&self) -> Result<(), arrow2::error::Error> {
        let reordered_indices: Vec<u64> = self.indices.reordered_indices.iter().map(|x| *x as u64).collect();

        super::io::write_reordering_map(reordered_indices, &self.data_dir)
    }
}
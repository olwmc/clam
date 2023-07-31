/// The `BatchedArrowReader` is the file interface this library uses to deal with
/// the Arrow IPC format and batched data.
/*
TODO: I need to decide on ONE (read: any) way to deal with uneven indices

Right now, if you have uneven indices (i.e. your last file has 10 fewer rows or whatever)
then `BatchedArrowReader::get` will silently fail because it is seeking to the wrong place
because the metadata size is smaller!
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
pub(crate) struct ArrowIndices {
    pub original_indices: Vec<usize>,
    pub reordered_indices: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct BatchedArrowReader<T: Number> {
    pub indices: ArrowIndices,

    // The directory where the data is stored
    data_dir: PathBuf,
    metadata: ArrowMetaData<T>,
    readers: RwLock<Vec<File>>,

    // We allocate a column of the specific number of bytes
    // necessary (type_size * num_rows) at construction to
    // lessen the number of vector allocations we need to do.
    // This might be able to be removed. Unclear.
    _col: RwLock<Vec<u8>>,

    // We'd like to associate this handle with a type, hence the phantomdata
    _t: PhantomData<T>,
    // Start Data map <Batch#, Start of Data>
    // start_points: HashMap<usize, u64>
    // let start_of_data = match start_points.get(filename) {
    //     Some(start) => start,
    //     None => &metadata.start_of_data,
    // }
}

impl<T: Number> BatchedArrowReader<T> {
    // TODO: Implement a "safe" constructor that actually goes through each metadata and doesn't just guess lol
    // We can read the metadata of many files fairly quickly if we assume static type size

    pub(crate) fn new(data_dir: &str, uneven_split: bool) -> Result<Self, Box<dyn Error>> {
        let path = PathBuf::from(data_dir);
        let (mut handles, reordered_indices) = process_data_directory(&path)?;

        // Load in the necessary metadata from the file
        let mut metadata = ArrowMetaData::<T>::try_from(&mut handles[0])?;

        // If we have an uneven split, then we need to read the final file's metadata and grab its start
        // of data
        if uneven_split {
            let length = handles.len() - 1;
            let last_metadata = ArrowMetaData::<T>::try_from(&mut handles[length])?;

            metadata.uneven_split_start_of_data = Some(last_metadata.start_of_message);
        }

        // Index information
        let original_indices: Vec<usize> = (0..metadata.cardinality * handles.len()).collect();
        let reordered_indices = match reordered_indices {
            Some(indices) => indices,
            None => original_indices.clone(),
        };

        Ok(BatchedArrowReader {
            data_dir: path,
            indices: ArrowIndices {
                reordered_indices,
                original_indices,
            },

            readers: RwLock::new(handles),
            _t: Default::default(),
            _col: RwLock::new(vec![0u8; metadata.row_size_in_bytes()]),
            metadata,
        })
    }

    pub(crate) fn get(&self, index: usize) -> Vec<T> {
        let resolved_index = self.indices.reordered_indices[index];
        self.get_column(resolved_index)
    }

    fn get_column(&self, index: usize) -> Vec<T> {
        let metadata = &self.metadata;

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

    pub(crate) fn write_reordering_map(&self) -> Result<(), Box<dyn Error>> {
        super::io::write_reordering_map(&self.indices.reordered_indices, &self.data_dir)
    }

    pub(crate) fn metadata(&self) -> &ArrowMetaData<T> {
        &self.metadata
    }
}

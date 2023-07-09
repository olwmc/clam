/*
(Oliver)
    This should probably be two files. This basically holds two major pieces of functionality:
        1. All of the necessary data structures, algos, and parsing stuff to read a specific subset of arrow files.
        2. The actual dataset implementation.

    It may be worth looking into separating out this stuff in case we ever need to reuse the existing arrow parsing
    functionality. Or not, I'm not sure if this stuff will /ever/ be reused. ¯\_(ツ)_/¯
*/

use crate::number::Number;
use arrow2::array::{PrimitiveArray, UInt64Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::read::{read_file_metadata, FileReader};
use arrow2::io::ipc::write::{FileWriter, WriteOptions};
use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::Buffer;
use arrow_format::ipc::MessageHeaderRef::RecordBatch;
use std::fs::{read_dir, DirEntry, File};
use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::mem;
use std::path::PathBuf;

// Arrow's file header has a certain length
const ARROW_MAGIC_OFFSET: u64 = 12;

#[derive(Debug)]
struct ArrowMetaData {
    // The offsets of the buffers containing the validation data and actual data
    buffers: Vec<Buffer>,

    // The file pointer offset corresponding to the beginning of the actual data
    start_of_message: u64,

    // Number of rows in the dataset (we assume each col. has the same number)
    num_rows: usize,

    // The size of the type of the dataset in bytes
    type_size: usize,

    // The cardinality of the dataset
    cardinality: usize,
}

impl ArrowMetaData {
    fn row_size_in_bytes(&self) -> usize {
        self.num_rows * self.type_size
    }
}

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
    // lessen the number of constructions we need to do.
    // This might be able to be removed. Unclear.
    _col: Vec<u8>,

    // We'd like to associate this handle with a type, hence the phantomdata
    _t: PhantomData<T>,
}

impl<T: Number, U: Number> BatchedArrowDataset<T, U> {
    pub fn new(data_dir: &str, metric: fn(&[T], &[T]) -> U) -> Self {
        // TODO: This has to be ordered somehow
        // TODO: Load in reordering metadata
        let (mut handles, reordered_indices) = Self::process_directory(&PathBuf::from(data_dir));

        // Load in the necessary metadata from the file
        let metadata = BatchedArrowDataset::<T, U>::extract_metadata(&mut handles[0]);

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

    pub fn process_directory(data_dir: &PathBuf) -> (Vec<File>, Option<Vec<usize>>) {
        let mut reordering = None;
        let files: Vec<DirEntry> = read_dir(data_dir).unwrap().map(|file| file.unwrap()).collect();

        if files.iter().any(|file| file.file_name() == "reordering.arrow") {
            reordering = Some(Self::get_reordered_indices(data_dir));
        }

        let handles: Vec<File> = files
            .iter()
            .filter(|file| file.file_name() != "reordering.arrow")
            .map(|file| File::open(file.path()).unwrap())
            .collect();

        (handles, reordering)
    }

    // TODO: Wrap this in a Result
    pub fn get(&mut self, index: usize) -> Vec<T> {
        self.get_column(index)
    }

    fn extract_metadata(reader: &mut File) -> ArrowMetaData {
        let type_size = mem::size_of::<T>();

        // We read part of this ourselves, so we need to skip past the arrow header.
        // 12 bytes for "ARROW1" + padding
        reader.seek(SeekFrom::Start(ARROW_MAGIC_OFFSET)).unwrap();

        // We then read the next four bytes, this contains a u32 which has the size of the
        // metadata
        let mut four_byte_buf = [0u8; 4];
        reader.read_exact(&mut four_byte_buf).unwrap();

        // Calculate the metadata length, and then calculate the data start point
        let meta_size = u32::from_ne_bytes(four_byte_buf);
        let mut data_start = ARROW_MAGIC_OFFSET + meta_size as u64;

        // Stuff is always padded to an 8 byte boundary, so we add the padding to the offset
        data_start += data_start % 8;

        // The +4 here is to skip past the continuation bytes ff ff ff ff
        data_start += 4;

        // Seek to the start of the actual data.
        // https://arrow.apache.org/docs/format/Columnar.html#encapsulated-message-format
        reader.seek(SeekFrom::Start(data_start)).unwrap();

        // Similarly, the size of the metadata for the block is also a u32, so we'll read it
        reader.read_exact(&mut four_byte_buf).unwrap();
        let block_meta_size = u32::from_ne_bytes(four_byte_buf);

        // We then actually parse the metadata for the block using flatbuffer. This gives us
        // many things but most notably is the offsets necessary for getting to a given column in
        // a file, as well as the number of rows each column has. This together allows us to read
        // a file.
        let mut meta_buf = vec![0u8; block_meta_size as usize];
        reader.read_exact(&mut meta_buf).unwrap();
        let message = arrow_format::ipc::MessageRef::read_as_root(meta_buf.as_ref()).unwrap();

        // Here we grab the nodes and buffers. Nodes = Row information, basically, and buffers are
        // explained here https://arrow.apache.org/docs/format/Columnar.html#buffer-listing-for-each-layout
        // In short, a buffer is a "piece of information". Be it the validity information or the
        // actual data itself.
        //
        // Here we extract the header and the recordbatch that is contained within it. This recordbatch has
        // all of the offset and row/column information we need to traverse the file and get arbitrary access.
        //
        // NOTE: We don't handle anything other than recordbatch headers at the moment.
        //
        // Most of this stuff here comes from the arrow_format crate. We're just extracting the information
        // from the flatbuffer we expect to be in the file.
        let header = message.header().unwrap().unwrap();

        // TODO (OWM): Get rid of this obviously
        let RecordBatch(r) = header else { panic!("Header does not contain record batch"); };

        // Nodes correspond to, in our case, row information for each column. Therefore nodes.len() is the number
        // of columns in the recordbatch and nodes[0].length() is the number of rows each column has (we assume
        // homogeneous column heights)
        let nodes = r.nodes().unwrap().unwrap();
        let cardinality: usize = nodes.len();
        let num_rows: usize = nodes.get(0).unwrap().length() as usize;

        // We then convert the buffer references to owned buffers. This gives us the offset corresponding to the
        // start of each column and the length of each column in bytes. NOTE (OWM): Do we need to store the length?
        // We don't seem to use it. NOTE (OWM): We could save some memory by not storing the validation buffer info.

        // TODO: Figure out if we can just store something like "validation_size" and "column_size" and just seek
        // to (column_size * n) + (validation_size * (n + 1)). NOTE: Is this necessary? Is the locality of the
        // buffer infos that big of a deal?
        let buffers: Vec<Buffer> = r
            .buffers()
            .unwrap()
            .unwrap()
            .iter()
            .map(|b| Buffer {
                offset: b.offset(),
                length: b.length(),
            })
            .collect();

        // We then grab the start position of the message. This allows us to calculate our offsets
        // correctly. All of the offsets in the buffers are relative to this point.
        let start_of_message: u64 = reader.stream_position().unwrap();

        ArrowMetaData {
            buffers,
            start_of_message,
            type_size,
            num_rows,
            cardinality,
        }
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

        // Skip past the validity bytes (our data is assumed to be non-nullable)
        self.readers[reader_index]
            .seek(SeekFrom::Start(
                // The data buffer's offset is the start of the actual data.
                self.metadata.start_of_message + data_buffer.offset as u64,
            ))
            .unwrap();

        // We then load the data of this row into the column data buffer
        self.readers[reader_index].read_exact(&mut self._col).unwrap();

        self._col
            .chunks(self.metadata.type_size)
            .map(|chunk| T::from_ne_bytes(chunk).unwrap())
            .collect()
    }

    #[allow(dead_code)]
    fn write_reordering_map(&self) -> Result<(), arrow2::error::Error> {
        // TODO: This is dogshit
        let reordered_indices = self.indices.reordered_indices.iter().map(|x| *x as u64).collect();
        let array = UInt64Array::from_vec(reordered_indices);

        let schema = Schema::from(vec![Field::new("Reordering", DataType::UInt64, true)]);

        let file = File::create(self.data_dir.join("reordering.arrow")).unwrap();
        let options = WriteOptions { compression: None };
        let mut writer = FileWriter::try_new(file, schema, None, options)?;
        let chunk = Chunk::try_new(vec![array.boxed()])?;

        writer.write(&chunk, None)?;
        writer.finish()?;

        Ok(())
    }

    // TODO: Migrate this to use our home grown parsing
    #[allow(dead_code)]
    fn get_reordered_indices(path: &PathBuf) -> Vec<usize> {
        // Load in the file
        let mut reader = File::open(path.join(PathBuf::from("reordering.arrow"))).unwrap();

        // Load in its metadata using arrow2
        let metadata = read_file_metadata(&mut reader).unwrap();
        let mut reader = FileReader::new(reader, metadata, None, None);

        // There's only one column, so we grab it
        let binding = reader.next().unwrap().unwrap();
        let column = &binding.columns()[0];

        // Array implements `Any`, so we can downcase it to a PrimitiveArray<u64> without any isssues, then just convert that to usize.
        // Unwrapping here is fine because we assume non-nullable
        column
            .as_any()
            .downcast_ref::<PrimitiveArray<u64>>()
            .unwrap()
            .iter()
            .map(|x| *x.unwrap() as usize)
            .collect()
    }
}

impl<T: Number, U: Number> super::Dataset<T, U> for BatchedArrowDataset<T, U> {
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
    use super::*;
    use crate::dataset::Dataset;

    #[test]
    fn grab_col_raw() {
        // Construct the batched reader
        let mut dataset: BatchedArrowDataset<u8, f32> =
            BatchedArrowDataset::new("/home/olwmc/current/data", crate::distances::u8::euclidean);

        let column: Vec<u8> = dataset.get(10_000_000);
        println!("{:?}", column);

        assert_eq!(dataset.cardinality(), 20_000_000);
    }

    #[test]
    fn test_reordering_map() {
        // Construct the batched reader
        let dataset: BatchedArrowDataset<u8, f32> =
            BatchedArrowDataset::new("/home/olwmc/current/data", crate::distances::u8::euclidean);

        dataset.write_reordering_map().unwrap();

        drop(dataset);

        let dataset: BatchedArrowDataset<u8, f32> =
        BatchedArrowDataset::new("/home/olwmc/current/data", crate::distances::u8::euclidean);

        assert_eq!(dataset.indices().len(), 20_000_000);
        assert_eq!(dataset.indices.reordered_indices[0..10], (0..10).collect::<Vec<usize>>());
    }
}

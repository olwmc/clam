use arrow2::io::ipc::write::Record;
use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::Buffer;
use arrow_format::ipc::MessageHeaderRef::RecordBatch;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::{fmt, mem};

use crate::number::Number;

use super::ARROW_MAGIC_OFFSET;

#[derive(Debug)]
pub struct MetadataParsingError(String);

impl fmt::Display for MetadataParsingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Error parsing metadata: {}", self.0)
    }
}

impl Error for MetadataParsingError {}

#[derive(Debug)]
pub struct ArrowMetaData<T: Number> {
    // The offsets of the buffers containing the validation data and actual data
    pub buffers: Vec<Buffer>,

    // The file pointer offset corresponding to the beginning of the actual data
    pub start_of_message: u64,

    // Number of rows in the dataset (we assume each col. has the same number)
    pub num_rows: usize,

    // The size of the type of the dataset in bytes
    pub type_size: usize,

    // The cardinality of the dataset
    pub cardinality: usize,

    // We store the type information to assure synchronization in the case of
    // independently constructed dataset and metadata
    _t: PhantomData<T>,
}

impl<T: Number> ArrowMetaData<T> {
    pub fn row_size_in_bytes(&self) -> usize {
        self.num_rows * self.type_size
    }

    pub fn try_from(reader: &mut File) -> Result<Self, Box<dyn Error>> {
        Self::extract_metadata(reader)
    }

    fn extract_metadata(reader: &mut File) -> Result<ArrowMetaData<T>, Box<dyn Error>> {
        let type_size = mem::size_of::<T>();

        // We read part of this ourselves, so we need to skip past the arrow header.
        // 12 bytes for "ARROW1" + padding
        reader.seek(SeekFrom::Start(ARROW_MAGIC_OFFSET))?;

        // We then read the next four bytes, this contains a u32 which has the size of the
        // metadata
        let mut four_byte_buf = [0u8; 4];
        reader.read_exact(&mut four_byte_buf)?;

        // Calculate the metadata length, and then calculate the data start point
        let meta_size = u32::from_ne_bytes(four_byte_buf);
        let mut data_start = ARROW_MAGIC_OFFSET + meta_size as u64;

        // Stuff is always padded to an 8 byte boundary, so we add the padding to the offset
        data_start += data_start % 8;

        // The +4 here is to skip past the continuation bytes ff ff ff ff
        data_start += 4;

        // Seek to the start of the actual data.
        // https://arrow.apache.org/docs/format/Columnar.html#encapsulated-message-format
        reader.seek(SeekFrom::Start(data_start))?;

        // Similarly, the size of the metadata for the block is also a u32, so we'll read it
        reader.read_exact(&mut four_byte_buf)?;
        let block_meta_size = u32::from_ne_bytes(four_byte_buf);

        // We then actually parse the metadata for the block using flatbuffer. This gives us
        // many things but most notably is the offsets necessary for getting to a given column in
        // a file, as well as the number of rows each column has. This together allows us to read
        // a file.
        let mut meta_buf = vec![0u8; block_meta_size as usize];
        reader.read_exact(&mut meta_buf)?;
        let message = arrow_format::ipc::MessageRef::read_as_root(meta_buf.as_ref())?;

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
        let header = message.header()?.ok_or(MetadataParsingError(
            "File contains no relevant header information".to_string(),
        ))?;

        // Header is of type MessageHeaderRef, which has a few variants. The only relevant (and valid) one
        // for us is the RecordBatch variant. Therefore, we reject all other constructions at the moment.
        let r = ({
            if let RecordBatch(r) = header {
                Ok(r)
            } else {
                Err(MetadataParsingError("Header does not contain record batch".to_string()))
            }
        })?;

        // Nodes correspond to, in our case, row information for each column. Therefore nodes.len() is the number
        // of columns in the recordbatch and nodes[0].length() is the number of rows each column has (we assume
        // homogeneous column heights)
        let nodes = r.nodes()?.ok_or(MetadataParsingError(
            "Header contains no node information and thus cannot be read".to_string(),
        ))?;
        let cardinality: usize = nodes.len();
        let num_rows: usize = nodes
            .get(0)
            .ok_or(MetadataParsingError(
                "Header contains no nodes and thus cannot be read".to_string(),
            ))?
            .length() as usize;

        // We then convert the buffer references to owned buffers. This gives us the offset corresponding to the
        // start of each column and the length of each column in bytes. NOTE (OWM): Do we need to store the length?
        // We don't seem to use it. NOTE (OWM): We could save some memory by not storing the validation buffer info.

        // TODO: Figure out if we can just store something like "validation_size" and "column_size" and just seek
        // to (column_size * n) + (validation_size * (n + 1)). NOTE: Is this necessary? Is the locality of the
        // buffer infos that big of a deal?
        let buffers: Vec<Buffer> = r
            .buffers()?
            .ok_or(MetadataParsingError(
                "Metadata contains no buffers and thus cannot be read".to_string(),
            ))?
            .iter()
            .map(|b| Buffer {
                offset: b.offset(),
                length: b.length(),
            })
            .collect();

        // We then grab the start position of the message. This allows us to calculate our offsets
        // correctly. All of the offsets in the buffers are relative to this point.
        let start_of_message: u64 = reader.stream_position()?;

        Ok(ArrowMetaData {
            buffers,
            start_of_message,
            type_size,
            num_rows,
            cardinality,
            _t: Default::default(),
        })
    }
}

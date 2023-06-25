use crate::number::Number;
use arrow_format::ipc::planus::ReadAsRoot;
use arrow_format::ipc::MessageHeaderRef::RecordBatch;
use arrow_format::ipc::{Buffer, /*FieldNode*/};
use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::mem;

const ARROW_MAGIC_OFFSET: u64 = 12;

pub struct IpcMetaData {
    //nodes: Vec<FieldNode>,
    buffers: Vec<Buffer>,
    start_of_message: u64,
    type_size: usize,
    num_rows: usize,
}

pub struct ArrowHandle<R: Read + Seek, T: Number> {
    metadata: IpcMetaData,
    reader: R,
    _t: PhantomData<T>,

    // We allocate a column of the specific number of bytes
    // necessary (type_size * num_rows) at construction to
    // lessen the number of constructions we need to do.
    _col: Vec<u8>,
}

impl<R: Read + Seek, T: Number> ArrowHandle<R, T> {
    pub fn new(mut handle: R) -> Self {
	let metadata = ArrowHandle::<R, T>::get_metadata(&mut handle);
        ArrowHandle {
	    reader: handle,
            _t: Default::default(),
	    _col: vec![0u8; ArrowHandle::<R,T>::get_row_size_in_bytes(&metadata)],
            metadata,
        }
    }

    fn get_row_size_in_bytes(metadata: &IpcMetaData) -> usize {
	//metadata.nodes[0].length as usize * metadata.type_size
	metadata.num_rows * metadata.type_size
    }

    // TODO: Wrap this in a Result
    pub fn get(&mut self, index: usize) -> Vec<T> {
        self.get_column(index)
    }

    fn get_metadata(reader: &mut R) -> IpcMetaData {
        let type_size = mem::size_of::<T>();

        // We read part of this ourselves, so we need to skip past the arrow header.
        // 12 bytes for "ARROW1" + padding
        reader.seek(SeekFrom::Start(ARROW_MAGIC_OFFSET)).unwrap();

        // We then read the next four bytes, this contains a u32 which has the size of the
        // metadata
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf).unwrap();

        // Calculate the metadata length, and then calculate the data start point
        let meta_size = u32::from_ne_bytes(buf);
        let mut data_start = ARROW_MAGIC_OFFSET + meta_size as u64;

	// Stuff is always padded to an 8 byte boundary, so we add the padding to the offset
        data_start += data_start % 8;

	// The +4 here is to skip past the continuation bytes ff ff ff ff
	data_start += 4;

        // Seek to the start of the actual data.
	// https://arrow.apache.org/docs/format/Columnar.html#encapsulated-message-format
        reader.seek(SeekFrom::Start(data_start)).unwrap();

        // Similarly, the size of the metadata for the block is also a u32, so we'll read it
        reader.read_exact(&mut buf).unwrap();
        let block_meta_size = u32::from_ne_bytes(buf);

        // We then actually parse the metadata for the block using flatbuffer. This gives us
        // many things but most notably is the offsets necessary for getting to a given column in
        // a file, as well as the number of rows each column has. This together allows us to read
        // a file.
        let mut buf = vec![0u8; block_meta_size as usize];
        reader.read_exact(&mut buf).unwrap();
        let message = arrow_format::ipc::MessageRef::read_as_root(buf.as_ref()).unwrap();

        // Here we grab the nodes and buffers. Nodes = Row information, basically, and buffers are
        // explained here https://arrow.apache.org/docs/format/Columnar.html#buffer-listing-for-each-layout
        // In short, a buffer is a "piece of information". Be it the validity information or the
        // actual data itself.
        //
        // Here we extract the header and the recordbatch that is contained within it.
        //
        // NOTE: We don't handle anything other than recordbatch headers at the moment.
        //
        // Most of this stuff here comes from the arrow_format crate
        let header = message.header().unwrap().unwrap();
        let RecordBatch(r) = header else { panic!("Header does not contain record batch"); };
        //let nodes = r.nodes().unwrap().unwrap()
        //    .iter()
        //    .map(|n| FieldNode {
        //        length: n.length(),
        //        null_count: n.null_count(),
        //    })
        //    .collect();
	let num_rows = r.nodes().unwrap().unwrap().get(0).unwrap().length() as usize;

        let buffers = r.buffers().unwrap().unwrap()
            .iter()
            .map(|b| Buffer {
                offset: b.offset(),
                length: b.length(),
            })
            .collect();

        // We then grab the start position of the message. This allows us to calculate our offsets
        // correctly. All of the offsets in the buffers are relative to this point.
        let start_of_message = reader.stream_position().unwrap();

        IpcMetaData {
            //nodes,
            buffers,
            start_of_message,
            type_size,
	    num_rows,
        }
    }

    fn get_column(&mut self, index: usize) -> Vec<T> {
        // This index keeps track which buffer we're on as we traverse through the nodes. There's
        // probably a nicer way of doing this by using like chain and take(2) or something but this
        // works for now.
        // let node = self.metadata.nodes[index];

        // Becuase we're limited to primitive types, we only have to deal with buffer 0 and
        // buffer 1 which are the validity and data buffers respectively. Therefore for every
	// index, there are two buffers associated with that column, the second of which is
	// the data buffer, hence the 2*i+1.
        let data_buffer = self.metadata.buffers[index * 2 + 1];

        // Skip past the validity bytes (our data is assumed to be non-nullable)
        self.reader
            .seek(SeekFrom::Start(
		// The data buffer's offset is the start of the actual data.
                self.metadata.start_of_message + data_buffer.offset as u64,
            ))
            .unwrap();

        // Grab the number of rows in this node.
        //let num_rows = node.length as usize;

        // We then load the data of this row into a buffer
        //let mut data = vec![0u8; self.metadata.type_size * num_rows];
        self.reader.read_exact(&mut self._col).unwrap();

        self._col.chunks(self.metadata.type_size)
            .map(|chunk| T::from_ne_bytes(chunk).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow2::io::ipc::read::read_file_metadata;
    use arrow2::io::ipc::read::FileReader;
    use std::fs::File;

    #[test]
    fn grab_col_raw() {
        // Spawn file handle
        let f = File::open("/home/oliver/base-0.arrow").unwrap();
        let mut handle = ArrowHandle::new(f);

        // Get the column and print it B^) (Arbitrary column access ? ? ?)
        let column: Vec<u8> = handle.get(1_000_000);
        println!("{:?}", column);
    }

    #[test]
    fn grab_col_arrow2() {
        let mut reader = File::open("/home/oliver/base-0.arrow").unwrap();
        let metadata = read_file_metadata(&mut reader).unwrap();
        let mut reader = FileReader::new(reader, metadata, None, None);

        println!("{:?}", reader.next().unwrap().unwrap().columns()[1_000_000]);
    }
}

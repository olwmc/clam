use crate::number::Number;
use arrow2::array::{PrimitiveArray, UInt64Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::read::{read_file_metadata, FileReader};
use arrow2::io::ipc::write::{FileWriter, WriteOptions};
use std::io::{Read, Seek, SeekFrom};
use std::error::Error;
use std::{
    ffi::OsString,
    fs::{read_dir, File},
    path::PathBuf,
};

use super::batched_reader::ArrowIndices;
use super::REORDERING_FILENAME;

pub fn process_data_directory(data_dir: &PathBuf) -> Result<(Vec<File>, Option<Vec<usize>>), Box<std::io::Error>> {
    let mut reordering = None;

    // Very annoying. We need to sort these files to maintain consistent loading. read_dir does not do this in any
    // consistent way. We will do this lexiographically.

    let mut filenames: Vec<OsString> = read_dir(data_dir)?

        // TODO: Owm how can we get around this unwrap?
        .map(|file| file.unwrap().file_name())
        .collect();

    filenames.sort();

    if filenames.iter().any(|file| file == REORDERING_FILENAME) {
        reordering = Some(read_reordering_map(data_dir));
    }

    let handles: Vec<File> = filenames
        .iter()
        .filter(|name| *name != REORDERING_FILENAME)
        .map(|name| File::open(data_dir.join(name)).unwrap())
        .collect();

    Ok((handles, reordering))
}

pub(crate) fn write_reordering_map(indices: &ArrowIndices, data_dir: &PathBuf) -> Result<(), Box<dyn Error>> {
    let reordered_indices: Vec<u64> = indices.reordered_indices.iter().map(|x| *x as u64).collect();

    let array: PrimitiveArray<u64> = UInt64Array::from_vec(reordered_indices);

    let schema = Schema::from(vec![Field::new("Reordering", DataType::UInt64, true)]);

    let file = File::create(data_dir.join(REORDERING_FILENAME))?;
    let options = WriteOptions { compression: None };
    let mut writer = FileWriter::try_new(file, schema, None, options)?;
    let chunk = Chunk::try_new(vec![array.boxed()])?;

    writer.write(&chunk, None)?;
    writer.finish()?;

    Ok(())
}

pub fn read_bytes_from_file<T: Number>(reader: &mut File, offset: u64, buffer: &mut Vec<u8>) -> Vec<T> {
    // Here's where we do the mutating
    // Skip past the validity bytes (our data is assumed to be non-nullable)
    reader
        .seek(SeekFrom::Start(
            // The data buffer's offset is the start of the actual data.
            offset,
        ))
        .unwrap();

    // We then load the data of this row into the column data buffer
    reader.read_exact(buffer).unwrap();

    buffer
        .chunks(std::mem::size_of::<T>())
        .map(|chunk| T::from_ne_bytes(chunk).unwrap())
        .collect()
}

// TODO: Migrate this to use our home grown parsing
fn read_reordering_map(path: &PathBuf) -> Vec<usize> {
    // Load in the file
    let mut reader = File::open(path.join(PathBuf::from(REORDERING_FILENAME))).unwrap();

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

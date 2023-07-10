use crate::number::Number;
use arrow2::array::{PrimitiveArray, UInt64Array};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::read::{read_file_metadata, FileReader};
use arrow2::io::ipc::write::{FileWriter, WriteOptions};
use std::io::{Read, Seek, SeekFrom};
use std::{
    ffi::OsString,
    fs::{read_dir, File},
    path::PathBuf,
};

use super::REORDERING_FILENAME;

pub fn process_directory(data_dir: &PathBuf) -> (Vec<File>, Option<Vec<usize>>) {
    let mut reordering = None;

    // Very annoying. We need to sort these files to maintain consistent loading. read_dir does not do this in any
    // consistent way. We will do this lexiographically.

    let mut filenames: Vec<OsString> = read_dir(data_dir)
        .unwrap()
        .map(|file| file.unwrap().file_name())
        .collect();

    filenames.sort();

    if filenames.iter().any(|file| file == REORDERING_FILENAME) {
        println!("Reordering file found!");
        reordering = Some(read_reordering_map(data_dir));
        println!("Loaded reordering file!");
    }

    let handles: Vec<File> = filenames
        .iter()
        .filter(|name| *name != REORDERING_FILENAME)
        .map(|name| File::open(name).unwrap())
        .collect();

    (handles, reordering)
}

// TODO: Migrate this to use our home grown parsing
#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn write_reordering_map(reordered_indices: Vec<u64>, data_dir: &PathBuf) -> Result<(), arrow2::error::Error> {
    let array = UInt64Array::from_vec(reordered_indices);

    let schema = Schema::from(vec![Field::new("Reordering", DataType::UInt64, true)]);

    let file = File::create(data_dir.join(REORDERING_FILENAME)).unwrap();
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

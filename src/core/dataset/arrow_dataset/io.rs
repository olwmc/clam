use std::{path::PathBuf, fs::{File, read_dir, DirEntry}};
use arrow2::array::{PrimitiveArray, UInt64Array};
use arrow2::io::ipc::read::{read_file_metadata, FileReader};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use arrow2::io::ipc::write::{FileWriter, WriteOptions};
use super::REORDERING_FILENAME;

pub fn process_directory(data_dir: &PathBuf) -> (Vec<File>, Option<Vec<usize>>) {
    let mut reordering = None;
    let files: Vec<DirEntry> = read_dir(data_dir).unwrap().map(|file| file.unwrap()).collect();

    if files.iter().any(|file| file.file_name() == REORDERING_FILENAME) {
        reordering = Some(read_reordering_map(data_dir));
    }

    let handles: Vec<File> = files
        .iter()
        .filter(|file| file.file_name() != REORDERING_FILENAME)
        .map(|file| File::open(file.path()).unwrap())
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
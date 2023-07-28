use crate::number::Number;
use std::error::Error;
use std::io::{Read, Seek, SeekFrom};
use std::{
    ffi::OsString,
    fs::{read_dir, File},
    path::{Path, PathBuf},
};

use std::fs::create_dir;

use arrow2::{
    array::{Float32Array, PrimitiveArray, UInt64Array},
    chunk::Chunk,
    datatypes::{DataType, DataType::Float32, Field, Schema},
    io::ipc::read::{read_file_metadata, FileReader},
    io::ipc::write::{FileWriter, WriteOptions},
};

use super::REORDERING_FILENAME;
use rand::{Rng, SeedableRng};
use uuid::Uuid;

pub type FilesAndReorderingMap = (Vec<File>, Option<Vec<usize>>);

/// Scans a given directory for batch files and returns their handles as well as an optional
/// set of reordered indices which are read from a specific file. If no reordering map is
/// present the function will return just the file handles and `None`.
///
/// ## Note on file read order
/// The files will be read based off of lexicographical order. Meaning if the order if your
/// data is important, the files should be named in such a way that earlier files will sort
/// earlier. The reasoning for this is that `read_dir` has no cross-platform consistent
/// ordering, thus to maintain the validity of the reordering map, it is necessary to
/// standardize the order in which files are read in.
///
/// # Arguments
/// `data_dir`: A directory pointing to a batched dataset
pub(crate) fn process_data_directory(data_dir: &PathBuf) -> Result<FilesAndReorderingMap, Box<std::io::Error>> {
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

pub(crate) fn write_reordering_map(reordered_indices: &[usize], data_dir: &Path) -> Result<(), Box<dyn Error>> {
    let reordered_indices: Vec<u64> = reordered_indices.iter().map(|x| *x as u64).collect();

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

pub(crate) fn read_bytes_from_file<T: Number>(reader: &mut File, offset: u64, buffer: &mut [u8]) -> Vec<T> {
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

fn read_reordering_map(path: &Path) -> Vec<usize> {
    // Load in the file
    let mut reader = File::open(path.join(PathBuf::from(REORDERING_FILENAME))).unwrap();

    // Load in its metadata using arrow2
    let metadata = read_file_metadata(&mut reader).unwrap();
    let mut reader = FileReader::new(reader, metadata, None, None);

    // There's only one column, so we grab it
    let binding = reader.next().unwrap().unwrap();
    let column = &binding.columns()[0];

    // `Array` implements `Any`, so we can downcase it to a PrimitiveArray<u64> without any isssues, then
    // just convert that to usize. Unwrapping here is fine because we assume non-nullable because
    // `write_reordering_map` writes as non-nullable.
    column
        .as_any()
        .downcast_ref::<PrimitiveArray<u64>>()
        .unwrap()
        .iter()
        .map(|x| *x.unwrap() as usize)
        .collect()
}

/// Returns the path of the newly created dataset
#[allow(dead_code)]
pub(crate) fn generate_batched_arrow_test_data(
    batches: usize,
    dimensionality: usize,
    cols_per_batch: usize,
    seed: Option<u64>,
) -> PathBuf {
    // Create a new uuid'd temp directory to store our batches in
    let path = std::env::temp_dir().join(format!("arrow-test-data-{}", Uuid::new_v4()));
    create_dir(path.clone()).unwrap();

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap());

    // Create our fields
    let fields = (0..cols_per_batch)
        .map(|x| Field::new(x.to_string(), Float32, false))
        .collect::<Vec<Field>>();

    // From those fields construct our schema
    let schema = Schema::from(fields);

    // For each batch, we need to create a file, create the write options, create the file writer to write
    // the arrow data, craete our arrays and finally construct our chunk and write it out.
    for batch_number in 0..batches {
        let file = File::create(path.join(format!("batch-{}.arrow", batch_number))).unwrap();
        let options = WriteOptions { compression: None };
        let mut writer = FileWriter::try_new(file, schema.clone(), None, options).unwrap();

        let arrays = (0..cols_per_batch)
            .map(|_| {
                Float32Array::from_vec((0..dimensionality).map(|_| rng.gen_range(0.0..100_000.0)).collect()).boxed()
            })
            .collect();

        let chunk = Chunk::try_new(arrays).unwrap();
        writer.write(&chunk, None).unwrap();
        writer.finish().unwrap();
    }

    // Return the path to our temp directory where the files are stored
    path
}

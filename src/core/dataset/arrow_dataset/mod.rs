const ARROW_MAGIC_OFFSET: u64 = 12;
const REORDERING_FILENAME: &str = "reordering.arrow";

pub mod dataset;
mod io;
mod metadata;
mod reader;
mod tests;

mod util;

// Arrow's file header has a certain length
const ARROW_MAGIC_OFFSET: u64 = 12;
const REORDERING_FILENAME: &str = "reordering.arrow";

mod io;
mod tests;
mod metadata;
pub mod dataset;

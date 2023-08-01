// The main user-facing dataset implementation
pub use dataset::BatchedArrowDataset;
mod dataset;

// IPC metadata information and parsing
mod metadata;

// IPC batch reader. The glue between individual arrow files
mod reader;

// Various file i/o helpers and utilities
mod io;

mod tests;
mod util;
use super::reader::BatchedArrowReader;
use crate::number::Number;
use std::error::Error;

#[derive(Debug)]
pub struct BatchedArrowDataset<T: Number, U: Number> {
    name: String,
    metric: fn(&[T], &[T]) -> U,
    metric_is_expensive: bool,
    reader: BatchedArrowReader<T>,
}

impl<T: Number, U: Number> BatchedArrowDataset<T, U> {
    pub fn new(
        data_dir: &str,
        name: String,
        metric: fn(&[T], &[T]) -> U,
        metric_is_expensive: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let reader = BatchedArrowReader::new(data_dir)?;
        Ok(Self {
            name,
            metric,
            metric_is_expensive,
            reader,
        })
    }

    pub fn get(&self, idx: usize) -> Vec<T> {
        self.reader.get(idx)
    }

    pub fn reorder(&self) {
        // Do reordery things
        self.reader.write_reordering_map().unwrap()
    }
}

impl<T: Number, U: Number> crate::dataset::Dataset<T, U> for BatchedArrowDataset<T, U> {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn cardinality(&self) -> usize {
        self.reader.indices.original_indices.len()
    }

    fn dimensionality(&self) -> usize {
        // We assume dimensionality is constant throughout the dataset
        self.reader.metadata().num_rows
    }

    fn is_metric_expensive(&self) -> bool {
        self.metric_is_expensive
    }

    fn indices(&self) -> &[usize] {
        &self.reader.indices.original_indices
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        (self.metric)(&self.get(left), &self.get(right))
    }

    fn query_to_one(&self, query: &[T], index: usize) -> U {
        (self.metric)(query, &self.get(index))
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.reader.indices.reordered_indices.swap(i, j);
    }

    fn set_reordered_indices(&mut self, indices: &[usize]) {
        self.reader.indices.reordered_indices = indices.to_vec();
    }

    fn get_reordered_index(&self, i: usize) -> usize {
        self.reader.indices.reordered_indices[i]
    }
}

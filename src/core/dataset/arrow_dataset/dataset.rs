use crate::number::Number;
use super::batched_reader::BatchedArrowReader;

impl<T: Number, U: Number> crate::dataset::Dataset<T, U> for BatchedArrowReader<T, U> {
    fn name(&self) -> String {
        format!("Batched Arrow Dataset : {}", self.data_dir.to_str().unwrap())
    }

    fn cardinality(&self) -> usize {
        self.indices.original_indices.len()
    }

    fn dimensionality(&self) -> usize {
        self.metadata.num_rows
    }

    fn is_metric_expensive(&self) -> bool {
        false
    }

    fn indices(&self) -> &[usize] {
        &self.indices.original_indices
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        (self.metric)(&self.get(left), &self.get(right))
    }

    fn query_to_one(&self, query: &[T], index: usize) -> U {
        (self.metric)(query, &self.get(index))
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.indices.reordered_indices.swap(i, j);
    }

    fn set_reordered_indices(&mut self, _indices: &[usize]) {
        todo!()
    }

    fn get_reordered_index(&self, _i: usize) -> usize {
        todo!()
    }
}
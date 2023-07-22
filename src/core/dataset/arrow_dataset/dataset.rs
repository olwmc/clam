use super::batched_reader::BatchedArrowReader;
use crate::number::Number;

impl<T: Number, U: Number> crate::dataset::Dataset<T, U> for BatchedArrowReader<T, U> {
    fn name(&self) -> String {
        format!("Batched Arrow Dataset : {:?}", self.data_dir)
    }

    fn cardinality(&self) -> usize {
        self.indices.original_indices.len()
    }

    fn dimensionality(&self) -> usize {
        // TODO: Need to make this work lmao
        self.metadata[0].num_rows
    }

    fn is_metric_expensive(&self) -> bool {
        self.metric_is_expensive
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

    fn set_reordered_indices(&mut self, indices: &[usize]) {
        self.indices.reordered_indices = indices.to_vec();
    }

    fn get_reordered_index(&self, i: usize) -> usize {
        self.indices.reordered_indices[i]
    }
}

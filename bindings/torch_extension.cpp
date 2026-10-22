// Python bindings for the ProteinSearch-GPU CUDA ops.
// Input validation, device guards and stream handling live next to each
// kernel in cuda/*.cu; this file only declares and exports them.

#include <torch/extension.h>

#include <vector>

namespace psg {

at::Tensor l2_normalize(const at::Tensor& input);
at::Tensor l2_normalize_(at::Tensor input);
at::Tensor cosine_similarity(const at::Tensor& query, const at::Tensor& database);
std::vector<at::Tensor> topk(const at::Tensor& similarities, int64_t k);
at::Tensor gather(const at::Tensor& database, const at::Tensor& indices);
at::Tensor smith_waterman(const at::Tensor& query_seqs, const at::Tensor& db_seqs,
                          const at::Tensor& query_lens, const at::Tensor& db_lens,
                          int64_t gap_open, int64_t gap_extend);
at::Tensor smith_waterman_batch(const at::Tensor& query_seqs,
                                const at::Tensor& db_seqs,
                                const at::Tensor& query_lens,
                                const at::Tensor& db_lens,
                                const at::Tensor& pair_indices,
                                int64_t gap_open, int64_t gap_extend);

}  // namespace psg

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;

  m.def("l2_normalize", &psg::l2_normalize, py::arg("input"),
        "Row-wise L2 normalization of a 2D CUDA tensor (float32/float16/bfloat16). "
        "Matches torch.nn.functional.normalize(input, dim=1).");
  m.def("l2_normalize_inplace", &psg::l2_normalize_, py::arg("input"),
        "In-place row-wise L2 normalization of a contiguous 2D CUDA tensor.");
  m.def("cosine_similarity", &psg::cosine_similarity, py::arg("query"),
        py::arg("database"),
        "Dot product of query [dim] or [num_queries, dim] with database "
        "[db_size, dim], returned as float32. Equals cosine similarity only "
        "when both inputs are L2-normalized (use l2_normalize first).");
  m.def("topk", &psg::topk, py::arg("similarities"), py::arg("k"),
        "Row-wise top-k (largest, sorted) of a float32 [num_queries, db_size] "
        "CUDA tensor for 1 <= k <= 32. Returns (values, int64 indices).");
  m.def("gather", &psg::gather, py::arg("database"), py::arg("indices"),
        "Gather database rows by int64 indices [num_queries, k] into "
        "[num_queries, k, dim]. Out-of-range indices (e.g. -1) give zero rows.");
  m.def("smith_waterman", &psg::smith_waterman, py::arg("query_seqs"),
        py::arg("db_seqs"), py::arg("query_lens"), py::arg("db_lens"),
        py::arg("gap_open") = 11, py::arg("gap_extend") = 1,
        "All-pairs Smith-Waterman scores (BLOSUM62, affine gaps: a gap of "
        "length L costs gap_open + (L - 1) * gap_extend) for int8 residue-index "
        "sequences. Returns float32 [num_queries, num_db].");
  m.def("smith_waterman_batch", &psg::smith_waterman_batch,
        py::arg("query_seqs"), py::arg("db_seqs"), py::arg("query_lens"),
        py::arg("db_lens"), py::arg("pair_indices"), py::arg("gap_open") = 11,
        py::arg("gap_extend") = 1,
        "Smith-Waterman scores for selected (query index, db index) pairs given "
        "as int64 [num_pairs, 2]. Returns float32 [num_pairs].");
}

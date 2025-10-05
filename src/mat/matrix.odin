/*
 Data types and basic helper functions for a few matrix types.
 -------------------------------------------------------------
 MATRIX_TYPES:
  - Dense, plain dense matrix stored column-major.
  - Block-sparse, under the BCSR format. Some optimizations for block-size = 1 (CSR).
  - Nested, matrix of multiple submatrices. Submatrices can be dense or sparse.
*/
package mat

Vector :: []f64

// wraps a vector with a block size, used in some routines that operate on BCSR matrices.
Block_Vector :: struct {
    vec: Vector,
    block_size: int,
}

Nested_Vector :: struct {
    []union{Vector, Block_Vector}
}

Sparsity :: struct {
 row_ptrs: []int,
 columns:  []int,
}

// Use for block sparse, and just plain sparse matrices.
Sparse_Mat :: struct {
    using sparsity: Sparsity,
    values: []f64,
    block_size: int,
}

// column-major.
Dense_Mat :: struct {
    values: [^]f64, // multi pointer as length can be computed.
    rows, columns: i32, // saves 8 bytes, can't go lower because of alignment.
}

// Use for block-structured matrices where block sizes are varying (block-sparse won't work).
Nested_Mat :: struct {
    mats: []union{Sparse_Mat, Dense_Mat},
    rows, columns: int,
}
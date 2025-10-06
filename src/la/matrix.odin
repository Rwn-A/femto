// SPDX-FileCopyrightText: 2025 Rowan Apps, Tor Rabien
// SPDX-License-Identifier: MIT
/*
 Data types and basic helper functions for a few matrix types.
 -------------------------------------------------------------
 MATRIX TYPES:
  - Dense, plain dense matrix stored column-major.
  - Block-sparse, under the BCSR format. Some optimizations for block-size = 1 (CSR).
  - Nested, matrix of multiple submatrices. Submatrices can be dense or sparse.

Heavy use of asserts, by design.
*/
package la

Vec :: []f64

// wraps a vector with a block size, used in some routines that operate on BCSR matrices.
Block_Vec :: struct {
    vec: Vec,
    block_size: int,
}

Nested_Vector :: struct {
    []Nestable_Vec,
}

Nestable_Vec :: union {
    Vec,
    Block_Vec,
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
    block_columns: int,
}

// column-major.
Dense_Mat :: struct {
    values: [^]f64, // multi pointer as length can be computed.
    rows, columns: i32, // saves 8 bytes, can't go lower because of alignment.
}

// Used in nested matrices to show that mathematically values exist here, but they are all zero.
Nil_Mat :: struct {
    rows, columns: int,
}

// Use for block-structured matrices where block sizes are varying (block-sparse won't work).
Nested_Mat :: struct {
    mats: []Nestable_Mat, //stored row-major
    rows, columns: int,
}

// Any matrix suitable inside a nested matrix.
Nestable_Mat :: union {
    Sparse_Mat,
    Dense_Mat,
    Nil_Mat,
}

// Creation

smat_from_sparsity :: proc(sparsity: Sparsity, block_size, block_columns: int, allocator := context.allocator) -> Sparse_Mat {
    return {
		sparsity = sparsity,
		values = make([]f64, len(sparsity.columns) * block_size * block_size, allocator),
		block_size = block_size,
		block_columns = block_columns,
	}
}

bvec_from_sparstiy :: proc(sparsity: Sparsity, block_size: int, allocator := context.allocator) -> Block_Vec {
    return {make(Vec, smat_block_rows(sparsity) * block_size, allocator), block_size}
}

dmat_create :: proc(#any_int rows, columns: i32) -> Dense_Mat {
    return {make([^]f64, rows * columns), rows, columns}
}

// The `mats` field must have an entry for each submatrix.
// We copy out the given slice, so it need not live past this call.
// `rows` and `columns` are how many submatrices exist in each direction.
nmat_create :: proc(rows, columns: int, mats: []Nestable_Mat, allocator := context.allocator) -> (nmat: Nested_Mat) {
    assert(rows > 0 && columns > 0, "Nested matrix must have positive dimensions.")
    assert(len(mats) == rows * columns, "Must pass in a matrix for each entry.")

    // validating the shape

    for sub_row in 0..<rows {
        expected_height := -1
        for sub_col in 0..<columns {
            idx := sub_row * columns + sub_col
            h, _ := mat_size(mats[idx])
            if expected_height == -1 {
                expected_height = h
            } else {
                assert(h == expected_height, "Mismatch rows in nested matrix creation.")
            }
        }
    }

    for sub_col in 0..<columns {
        expected_width := -1
        for sub_row in 0..<rows {
            idx := sub_row * columns + sub_col
            _, w := mat_size(mats[idx])
            if expected_width == -1 {
                expected_width = w
            } else {
                assert(w == expected_width, "Mismatch columns in nested matrix creation.")
            }
        }
    }

    nmat = Nested_Mat{ make([]Nestable_Mat, rows * columns, allocator), rows, columns }
    copy(nmat.mats, mats)
    return
}

// Shapes

smat_block_rows :: #force_inline proc(sparsity: Sparsity) -> int {
    return len(sparsity.row_ptrs) - 1
}

// Returns the total rows/columns, not the block rows/columns.
smat_size :: #force_inline proc(smat: Sparse_Mat) -> (rows, columns: int) {
    return smat_block_rows(smat) * smat.block_size, smat.block_columns * smat.block_size
}

dmat_size :: #force_inline proc(dmat: Dense_Mat) -> (rows, columns: int) {
    return int(dmat.rows), int(dmat.columns)
}

nilmat_size :: #force_inline proc(nil_mat: Nil_Mat) -> (rows, columns: int) {
    return nil_mat.rows, nil_mat.columns
}

nmat_size :: proc(nmat: Nested_Mat) -> (rows, columns: int) {
    if len(nmat.mats) == 0 { return 0, 0 }

    for block_row in 0..<nmat.rows {
        h, _ := mat_size(nmat.mats[block_row * nmat.columns])
        rows += h
    }

    for block_col in 0..<nmat.columns {
        _, w := mat_size(nmat.mats[block_col])
        columns += w
    }

    return
}

nestable_mat_size :: proc(mat: Nestable_Mat) -> (rows, columns: int) {
    switch inner in mat {
        case Dense_Mat: return mat_size(inner)
        case Sparse_Mat: return mat_size(inner)
        case Nil_Mat: return mat_size(inner)
    }
    return 0, 0
}

mat_size :: proc {
    smat_size,
    dmat_size,
    nilmat_size,
    nmat_size,
    nestable_mat_size,
}

// Indexing

// Specific value index within a specific block.
smat_index_of_value :: proc(mat: Sparse_Mat, block_row, block_col, local_row, local_col: int) -> int {
    assert(local_row < mat.block_size && local_col < mat.block_size, "local_row or local_col are out of bounds of block size.")
	block_idx := smat_index_of_block(mat, block_row, block_col)
	return (block_idx * mat.block_size * mat.block_size) + (local_col * mat.block_size) + local_row
}

// Index into the columns array for a specific block, cannot index values directly must account for block width.
smat_index_of_block :: proc(sp: Sparsity, block_row, block_col: int) -> int {
    for i in sp.row_ptrs[block_row] ..< sp.row_ptrs[block_row + 1] {
		if sp.columns[i] == block_col {return i}
	}
	panic("Qeuried block that was not defined in sparsity.")
}

bvec_index_of :: #force_inline proc(vec: Block_Vec, block_row, local_row: int) -> int {
    return (block_row * vec.block_size) + local_row
}

dmat_index_of :: #force_inline proc(mat: Dense_Mat, #any_int row, col: i32) -> int {
    assert(row < mat.rows && col < mat.columns)
	return int(col * mat.rows + row)
}

nmat_index_of :: #force_inline proc(mat: Nested_Mat, row, col: int) -> int {
    assert(row < mat.rows && col < mat.columns)
    return (col * mat.rows + row)
}

// Extractions


// Returns the block as a slice of the given block vector
bvec_extract_block :: #force_inline proc(vec: Block_Vec, block_row: int) -> Vec {
    return vec.vec[block_row * vec.block_size:][:vec.block_size]
}

// Scatters the entries of v into provided output vectors.
bvec_scatter :: proc(vec: Block_Vec, outputs: []Vec) {
    assert(len(outputs) == vec.block_size)
	for block_row in 0 ..< len(vec.vec) / vec.block_size {
		for local_row in 0 ..< vec.block_size {
			global_idx := bvec_index_of(vec, block_row, local_row)
			outputs[local_row][block_row] = vec.vec[global_idx]
		}
	}
}

// Interleaves the entrys of vectors into the output vector.
bvec_gather :: proc(vectors: []Vec, output: Block_Vec) {
	assert(len(vectors) == output.block_size)
	for vec in vectors {assert(len(output.vec) == len(vec) * output.block_size)}

	for block_row in 0 ..< len(output.vec) / output.block_size {
		for local_row in 0 ..< output.block_size {
			output.vec[bvec_index_of(output, block_row, local_row)] = vectors[local_row][block_row]
		}
	}
}



// Returns the block as a dense matrix, block index can be gotten from `bsmat_index_of_block`.
// does not copy the data, matrix is a view into the existing sparse matrix.
smat_extract_block :: #force_inline proc(mat: Sparse_Mat, block_idx: int) -> Dense_Mat {
    return {rows = i32(mat.block_size), columns = i32(mat.block_size), values = &mat.values[block_idx * mat.block_size * mat.block_size]}
}

// Fills the backing_vector with diagonal entries, returns a block vector view into that backing.
smat_extract_diagonal :: proc(mat: Sparse_Mat, backing_vector: Vec) -> Block_Vec {
    assert(len(backing_vector) == (smat_block_rows(mat) * mat.block_size))
    output := Block_Vec{backing_vector, mat.block_size}
	for block_row in 0 ..< smat_block_rows(mat) {
		for local_row in 0 ..< mat.block_size {
	        backing_vector[bvec_index_of(output, block_row, local_row)] = mat.values[smat_index_of_value(mat, block_row, block_row, local_row, local_row)]
		}
	}
	return output
}



// Destruction

dmat_destroy :: proc(mat: Dense_Mat, allocator := context.allocator) {
    free(mat.values, allocator)
}

// Does not delete sparsity pattern
smat_destroy :: proc(mat: Sparse_Mat, allocator := context.allocator) {
    delete(mat.values, allocator)
}

// Does not destroy sub matrices
nmat_destroy :: proc(mat: Nested_Mat, allocator := context.allocator) {
    delete(mat.mats, allocator)
}

sparsity_destroy :: proc(sparsity: Sparsity, allocator := context.allocator) {
    delete(sparsity.row_ptrs, allocator)
    delete(sparsity.columns, allocator)
}


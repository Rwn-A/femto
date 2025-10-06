// SPDX-FileCopyrightText: 2025 Rowan Apps, Tor Rabien
// SPDX-License-Identifier: MIT
/*
 Basic wrapper over infra contended writer for sparse linear systems.
 --------------------------------------------------------------------
 Works for both explicitly assembled matrices, and matrix-free methods.

 Expected usage:
 ```
  lsw_data := lsw_create_assembled(my_matrix, my_rhs)

  parallel_for_contended(prt, {0, 10}, lsw_outputs(lsw_data)[:], Data, proc(data: Data, range: Range, w: ^Contended_Writer(f64)){
    lsw := LS_Writer{data.lsw_data, w}

    // write to very frist matrix entry, and rhs entry the value 1.
    ls_insert_value(lsw, 0, 0, 0, 0, 1, 1)
  })
 ```
*/
package la

import "../infra"

Mat_Write_Mode :: enum {
    Assembled,
    Free,
}

// used to index the output arrays in contended writer.
LS_Output :: enum {
    Mat = 0,
    Rhs = 1,
}

// Expected to exist outside a parallel for scope.
LS_Writer_Data :: struct {
    mode: Mat_Write_Mode,
    rhs: Block_Vec,
    mat: Sparse_Mat, // only active in assembled mode.
    x_vec: Block_Vec, // only active in mat free mode. y = A*x
    y_vec: Block_Vec, // only active in mat free mode. this is where we output `.Mat` entries to.
}

// Created within each thread of a parallel for.
LS_Writer :: struct {
    using data: LS_Writer_Data,
    cw: ^infra.Contended_Writer(f64),

}

// Creates writer data for a matrix-free form. `input` is your x vector in y = A * x, the `output` vector is the y.
lsw_create_mat_free :: #force_inline proc(rhs, input, output: Block_Vec) -> LS_Writer_Data {
    return { mode = .Free, rhs = rhs, x_vec = input, y_vec = output }
}

// Creates writer data for a matrix explicitly in memory.
lsw_create_assembled :: #force_inline proc(mat: Sparse_Mat, rhs: Block_Vec) -> LS_Writer_Data {
    return { mode = .Assembled, rhs = rhs, mat = mat }
}

// pass in to `parallel_for_contended` outputs field.
lsw_outputs :: #force_inline proc(lsw_data: LS_Writer_Data) -> [LS_Output][]f64 {
    return {.Mat = lsw_data.mat.values if lsw_data.mode == .Free else lsw_data.y_vec.vec, .Rhs = lsw_data.rhs.vec}
}

// Expects to be called within a `parallel_for_contended`. Multiple writes to the same place will accumulate.
lsw_insert_value :: proc(lsw: ^LS_Writer, block_row, block_col, local_row, local_col: int, mat_value: f64, rhs_value: f64) {
    assert(lsw != nil && lsw.mode != nil, "Got a nil linear system writer.")

    if rhs_value != 0 {
        infra.contended_write(lsw.cw, int(LS_Output.Rhs), bvec_index_of(lsw.rhs, block_row, local_row), rhs_value)
    }

    if mat_value == 0 { return }

    switch lsw.mode {
        case .Assembled:
            infra.contended_write(lsw.cw, int(LS_Output.Mat), smat_index_of_value(lsw.mat, block_row, block_col, local_row, local_col), mat_value)
        case .Free:
            input_index := bvec_index_of(lsw.x_vec, block_col, local_col)
            output_index := bvec_index_of(lsw.y_vec, block_row, local_row)
            infra.contended_write(lsw.cw, int(LS_Output.Mat), output_index, mat_value * lsw.x_vec.vec[input_index])
        case: unreachable()
    }
}
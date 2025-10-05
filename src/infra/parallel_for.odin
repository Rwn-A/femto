/*
 Simple parallel for implementation with support for possibly contended writes.
 ------------------------------------------------------------------------------
 USE CASES:
  - Work is roughly the same per iteration.
  - Designed for CPU bound tasks.
  - Some writes may conflict across threads.
*/
package infra

import "base:intrinsics"
import "base:runtime"
import "core:sync"
import "core:thread"
import "core:mem/virtual"

import "core:testing"
import "core:time"
import "core:log"

// TODO: Allow for flushing mid loop to lower peak memory at the cost of come CPU time.
// TODO: Clean up some false sharing, especially in contended writer.
// TODO: Could add a spin-lock, if the barriers really seem to matter.
// TODO: opt-in work stealing, if we need it, probably will for p-adataptible FEM

// Disable multi-threading, no matter how many threads you ask for it always just uses 1.
USE_THREADS :: #config(USE_THREADS, true)

// Correct for most systems, not sure how to get it programtically.
CACHE_LINE_SIZE :: 64

// [start, end)
Range :: struct {
	start, end: int,
}

// Do not copy after initialization
Parallel_Runtime :: struct {
	allocator:            runtime.Allocator,
	spawned_threads:      []^thread.Thread, // the threads internal id `tid` is the index into this array, main thread id is 1 + the last index here.
	owner_thread_id:      int, // OS thread id, not the internal id.
	total_threads:        int,
	// sync stuff, any below fields are sync primitives or atomic.
	start_barrier:        sync.Barrier,
	done_barrier:         sync.Barrier,
	should_shutdown:      bool,
	// state updated per call to parallel_for_*
	work_ranges:          []Range, //indexed by tid
	user_data:            rawptr,
	user_proc:            proc(data: rawptr, range: Range),
}

// State for contended writes, created during a call to `parallel_for_contended`.
Contended_Writer :: struct(E: typeid) {
	outputs: [][]E,
	owned_indices: [][]Range, //per thread, per buffer
	orphan_buffers: [][dynamic]Orphaned_Entry(E), //per thread
	mode: Write_Mode,
}

// A entry from `contended_write` that was not able to be written immediately.
Orphaned_Entry :: struct(E: typeid) {
	value: E,
	index: int,
	output_index: int,
}

// Controls how entries from `contended_write` are added to the output arrays.
Write_Mode :: enum {
	Accumulate,
	Decumulate,
}


// Set on initialization, value from 0 -> total_threads - 1, useful for accessing thread specific resources from the data parameter.
// Also used by contended writer to know what thread is calling it.
// modifying this is a big no-no.
@(thread_local)
tid: int

// Sets up the runtime for subsequent calls, spawns `helper_threads` total threads, cannot copy runtime after this.
parallel_runtime_init :: proc(
	rt: ^Parallel_Runtime,
	helper_threads: int,
	allocator := context.allocator,
	loc := #caller_location,
) {
	assert(helper_threads >= 1, "Must use 1 or more helper threads.", loc)
	context.allocator = allocator

	rt.allocator = allocator
	rt.total_threads = helper_threads + 1 when USE_THREADS else 1
	rt.owner_thread_id = sync.current_thread_id()

	rt.work_ranges = make([]Range, rt.total_threads)

	sync.barrier_init(&rt.start_barrier, rt.total_threads)
	sync.barrier_init(&rt.done_barrier, rt.total_threads)
	when USE_THREADS {
		rt.spawned_threads = make([]^thread.Thread, helper_threads)
		for &spawned_thread, i in rt.spawned_threads {
			spawned_thread = thread.create_and_start_with_poly_data2(rt, i + 1, thread_proc) // tid 0 is main hence + 1
		}
	}

	//main thread, 0 tid, allows for multiple runtimes wiht the same main.
	tid = 0

	thread_proc :: proc(rt: ^Parallel_Runtime, thread_index: int) {
		tid = thread_index
		for {
			sync.barrier_wait(&rt.start_barrier)

			//we might have been woken up solely to exit this loop, hence check first.
			if sync.atomic_load(&rt.should_shutdown) {break}

			//do said work
			work_range := rt.work_ranges[tid]

			//make sure we actually have some work to do.
			if work_range.start < work_range.end {
				rt.user_proc(rt.user_data, work_range)
			}

			//signal we are done
			sync.barrier_wait(&rt.done_barrier)
		}
	}
}

// Returns the part of a range a specifc thread would be tasked with if run with `parallel_for_*.`
parallel_runtime_partition :: proc(prt: ^Parallel_Runtime, range: Range, tid: int, loc := #caller_location) -> Range {
	assert(tid >= 0 && tid < prt.total_threads, "tid is out of bounds.", loc)
	assert(prt_is_initialized(prt), "Runtime must be initialized.", loc)

	work := range.end - range.start
	chunk_size := work / prt.total_threads
	chunk_start := range.start + tid * chunk_size
	chunk_end := range.start + (tid + 1) * chunk_size

	// last tid handles last chunk + remainder.
	if tid == prt.total_threads - 1 {chunk_end = range.end}

	return {chunk_start, chunk_end}
}

parallel_runtime_shutdown :: proc(prt: ^Parallel_Runtime, loc := #caller_location) {
	assert(prt_is_initialized(prt), "Runtime must be initialized.", loc)
	assert(sync.current_thread_id() == prt.owner_thread_id, "Must call from thread that created runtime.", loc)

	context.allocator = prt.allocator
	sync.atomic_store(&prt.should_shutdown, true)

	sync.barrier_wait(&prt.start_barrier)

	thread.join_multiple(..prt.spawned_threads)
	for &helper in prt.spawned_threads {thread.destroy(helper)}

	delete(prt.work_ranges)
	delete(prt.spawned_threads)
}


// Runtime polymorphic parallel for, use `spin_lock = true` if you plan on calling two or more loops back to back.
// The body procedure should process all entries in the range it's given.
parallel_for_untyped :: proc(
	prt: ^Parallel_Runtime,
	range: Range,
	data: rawptr,
	body: proc(data: rawptr, range: Range),
	loc := #caller_location,
) {
	assert(sync.current_thread_id() == prt.owner_thread_id, "Must call from thread that created runtime.", loc)
	assert(prt_is_initialized(prt), "Runtime must be initialized.", loc)
	prt.user_data = data
	prt.user_proc = body
	for &work_range, tid in prt.work_ranges {work_range = parallel_runtime_partition(prt, range, tid)}

	sync.barrier_wait(&prt.start_barrier)

	main_range := prt.work_ranges[tid]
	if main_range.start < main_range.end {prt.user_proc(prt.user_data, main_range)}

	// signal main is finished and wait for everyone else
	sync.barrier_wait(&prt.done_barrier)
}

// generic wrapper over untyped parallel for, see `parallel_for_untyped` for more docs.
parallel_for :: proc(
	prt: ^Parallel_Runtime,
	range: Range,
	data: $T,
	body: proc(data: T, range: Range),
	loc := #caller_location,
) {
	Wrapped_Data :: struct {
		data: T,
		body: proc(data: T, range: Range),
	}

	wrapped_data := Wrapped_Data{data, body}

	untyped_body :: proc(data: rawptr, range: Range) {
		wrapped_data := cast(^Wrapped_Data)(data)
		wrapped_data.body(wrapped_data.data, range)
	}

	parallel_for_untyped(prt, range, &wrapped_data, untyped_body, loc)
}


// Same as `parallel_for` but each body procedure comes with a writer.
// Using `contended_write` you can write into any index of any of the arrays in `outputs` without worrying about thread safety.
// The allocator is required to set up the thread safety mechanisms, that allocator must be thread safe.
// For best performance, the first chunk of `range` should write to the first chunk of each output buffer.
// If writes are all over the place, or mostly unaligned between input output ranges,
// memory usage and performance will suffer.
parallel_for_contended :: proc(
	prt: ^Parallel_Runtime,
	range: Range,
	outputs: [][]$E,
	data: $T,
	body: proc(data: T, range: Range, w: ^Contended_Writer(E)),
	allocator: runtime.Allocator,
	mode := Write_Mode.Accumulate,
	loc := #caller_location,
) {
	assert(len(outputs) != 0, "Expected atleast 1 output array")
	for output in outputs {assert(len(output) != 0, "Output arrays must have non-zero length.")}

	Wrapped_Data :: struct {
		w:    ^Contended_Writer(E),
		data: T,
		body: proc(data: T, range: Range, w: ^Contended_Writer(E)),
	}

	// use main thread arena for some of these allocations.
	// Its freed by the parallel runtime after `untyped_body`.
	context.allocator = allocator

	w: Contended_Writer(E)
	w.outputs = outputs
	w.mode = mode
	w.orphan_buffers = make([][dynamic]Orphaned_Entry(E), prt.total_threads)
	w.owned_indices = make([][]Range, prt.total_threads)

	for &arr in w.orphan_buffers { arr = make([dynamic]Orphaned_Entry(E)) }

	// each thread has an owned range per output array.
	for &output_index_ranges, tid in w.owned_indices {
		output_index_ranges = make([]Range, len(outputs))
		for &range, output_index in w.owned_indices[tid] {
			range = parallel_runtime_partition(prt, {0, len(w.outputs[output_index])}, tid)
		}
	}

	defer {
    	for buffer in w.orphan_buffers { delete(buffer) }
    	delete(w.orphan_buffers)
    	for ranges in w.owned_indices { delete(ranges) }
    	delete(w.owned_indices)
	}

	wrapped_data := Wrapped_Data{&w, data, body}

	untyped_body :: proc(data: rawptr, range: Range) {
		wrapped_data := cast(^Wrapped_Data)(data)
		wrapped_data.body(wrapped_data.data, range, wrapped_data.w)
	}

	parallel_for_untyped(prt, range, &wrapped_data, untyped_body, loc)

	flush :: proc(data: rawptr, range: Range) {
		wrapped_data := cast(^Wrapped_Data)(data)
		for buffer in wrapped_data.w.orphan_buffers {
			for entry in buffer {
				if !in_range(entry.index, wrapped_data.w.owned_indices[tid][entry.output_index]) {continue}
				unchecked_write(wrapped_data.w, entry.output_index, entry.index, entry.value)
			}
		}
	}

	parallel_for_untyped(prt, range, &wrapped_data, flush)
}

// Expected to be called within the body procedure of a `parallel_for_contended`.
contended_write :: proc(cw: ^Contended_Writer($E), output_index, value_index: int, value: E) {
	if in_range(value_index, cw.owned_indices[tid][output_index]) {
		unchecked_write(cw, output_index, value_index, value)
	}else {
		append(&cw.orphan_buffers[tid], Orphaned_Entry(E){value, value_index, output_index})
	}
}

@(private="file")
unchecked_write :: #force_inline proc(cw: ^Contended_Writer($E), output_index, value_index: int, value: E) {
	val := &cw.outputs[output_index][value_index]
	switch cw.mode {
		case .Accumulate: val^ += value
		case .Decumulate: val^ -= value
	}
}

in_range :: #force_inline proc(val: int, range: Range) -> bool {return val >= range.start && val < range.end}
prt_is_initialized :: #force_inline proc(prt: ^Parallel_Runtime) -> bool {return prt != nil && prt.total_threads >= 1}

// Non exhaustive, usage examples / basic tests to make sure its not deadlock city.

@(test)
test_standard_loop :: proc(t: ^testing.T) {
	rt: Parallel_Runtime
	parallel_runtime_init(&rt, 3)

	arr: [1000]int
	for &x in arr {x = 1}

	parallel_for(&rt, {0, 1000}, arr[:], proc(arr: []int, r: Range) {for i in r.start..<r.end{arr[i] += 1}})

	for i in 0..<1000 { testing.expect_value(t, arr[i], 2) }

	parallel_runtime_shutdown(&rt)
}

@(test)
test_contended_write :: proc(t: ^testing.T) {
	rt: Parallel_Runtime
	parallel_runtime_init(&rt, 3)

	out: [100]int

	body :: proc(dummy: int, r: Range, w: ^Contended_Writer(int)) {
		for i in r.start..<r.end {
			target_index := i % 100
			contended_write(w, 0, target_index, 1)
		}
	}
	parallel_for_contended(&rt, {0, 10_000}, [][]int{out[:]}, 0, body, context.allocator)

	sum := 0
	for x in out { sum += x }

	testing.expect_value(t, sum, 10_000)

	parallel_runtime_shutdown(&rt)
}
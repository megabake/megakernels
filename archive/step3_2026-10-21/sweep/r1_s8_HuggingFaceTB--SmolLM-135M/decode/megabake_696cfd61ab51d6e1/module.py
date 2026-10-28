import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cute.tensor import _Tensor

from megabake.runtime import (
    counting_grid_arrive,
    counting_grid_barrier,
    counting_grid_wait,
    prefetch_gemm_weights,
    prefetch_l1,
    trap_if,
)
from megabake.bodies.gemm import (
    HopperWgmmaGemmPersistentKernel,
    build_table_descriptors,
    table_descriptors,
    union_shared_storage,
)


@cute.jit
def _gemm_slot(
    addresses: cute.Tensor,
    gemm_table: cute.Tensor,
    gemm: cutlass.Int32,
    column: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    layout: cutlass.Constexpr,
    alignment: cutlass.Constexpr,
):
    # A fused GEMM epilogue's tensor, from its GEMM row's slot column.
    address = addresses[gemm_table[gemm, column]]
    if cutlass.const_expr(dtype is cutlass.Boolean):
        storage = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int8, address, cute.AddressSpace.gmem, assumed_align=alignment
            ),
            layout,
        )
        return _Tensor(storage, dtype=cutlass.Boolean)
    return cute.make_tensor(
        cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=alignment),
        layout,
    )

@cute.jit
def _epilogue_0_1(accumulator, accumulator_1, o0, o1, tensor_bindings):
    return ((((tensor_bindings[0][0, 0, o1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o1) < 576 else cutlass.BFloat16(0))).to(cutlass.Float32) + ((accumulator).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)

@cute.jit
def _epilogue_2_1(accumulator, accumulator_1, o0, o1, tensor_bindings):
    return (((((((accumulator).to(cutlass.BFloat16)).to(cutlass.Float32) / (cute.exp((-(((accumulator).to(cutlass.BFloat16)).to(cutlass.Float32)))) + (cutlass.Int64(1)).to(cutlass.Float32)))).to(cutlass.BFloat16)).to(cutlass.Float32) * ((accumulator_1).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)

@cute.jit
def _epilogue_3_1(accumulator, accumulator_1, o0, o1, tensor_bindings):
    return ((((tensor_bindings[0][0, 0, o1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o1) < 576 else cutlass.BFloat16(0))).to(cutlass.Float32) + ((accumulator).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)

_gemm_0 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 16),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
    splits=1,
    bias_features=0,
    bias_along_m=True,
    direct_store=True,
    epilogues=(_epilogue_0_1, ),
    stream_stages=8,
)
_gemm_1 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 16),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
    splits=1,
    bias_features=0,
    bias_along_m=True,
    direct_store=True,
    stream_stages=8,
)
_gemm_2 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (128, 16),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
    splits=1,
    bias_features=0,
    bias_along_m=True,
    direct_store=True,
    epilogues=(_epilogue_2_1, ),
    paired=True,
    stream_stages=8,
)
_gemm_3 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 16),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
    splits=1,
    bias_features=0,
    bias_along_m=True,
    direct_store=True,
    epilogues=(_epilogue_3_1, ),
    stream_stages=8,
)
_gemm_4 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 16),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
    splits=1,
    bias_features=0,
    bias_along_m=True,
    direct_store=True,
    stream_stages=8,
)


@cute.jit
def _slot(
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    phase: cutlass.Int32,
    column: cutlass.Constexpr,
    dtype: cutlass.Constexpr,
    layout: cutlass.Constexpr,
    alignment: cutlass.Constexpr,
):
    # The host appends each phase's resolved slot addresses to the table.
    address = addresses[1345 + phase * 11 + column]
    if cutlass.const_expr(dtype is cutlass.Boolean):
        # Boolean tensors are byte-backed, matching DLPack-bound tensors.
        storage = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int8, address, cute.AddressSpace.gmem, assumed_align=alignment
            ),
            layout,
        )
        return _Tensor(storage, dtype=cutlass.Boolean)
    return cute.make_tensor(
        cute.make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=alignment),
        layout,
    )


@cute.struct
class _Phase2Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase10Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 2048],
        16
    ]


@cute.union
class _WorkerScratch:
    phase_2: _Phase2Scratch
    phase_10: _Phase10Scratch

@cute.kernel
def megabake_kernel(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    work_iterators: tuple[cute.Tensor, ...],
    dynamic_addresses: cutlass.Int64,
    dynamic_slot_table: cute.Tensor,
    phase_floats: cute.Tensor,
    phase_counters: cute.Tensor,
    gemm_table: cute.Tensor,
    stream_table: cute.Tensor,
    tma_descriptors: cute.Tensor,
    gemm_values,
):
    _, _, _block = cute.arch.block_idx()
    _thread, _, _ = cute.arch.thread_idx()
    # Every CTA reads the call counter before the first barrier; block 0
    # advances it after that barrier, for the next launch on this stream.
    _call = ready[workers + 2]
    _ordinal = _call * cutlass.Int32(153)
    _worker_scratch = cutlass.utils.SmemAllocator().allocate(_WorkerScratch)
    _scratch_2 = _worker_scratch.phase_2.body.get_tensor(cute.make_layout(8))
    _scratch_10 = _worker_scratch.phase_10.body.get_tensor(cute.make_layout(2048))
    if _thread == 0:
        _worker_scratch.phase_2.barrier_0[0] = 0
    cute.arch.sync_threads()

    _gemm_scratch = cutlass.utils.SmemAllocator().allocate(
        union_shared_storage((_gemm_0, _gemm_1, _gemm_2, _gemm_3, _gemm_4, ))
    )

    @cute.jit
    def _body_0(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[2]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 576
            remainder_0 = remainder_0 // 576
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            trap_if(load_0_0 < 0 or load_0_0 >= 49152)
            load_1_0 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2_0] if 0 <= (o2_0) < 576 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 576
            remainder_1 = remainder_1 // 576
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            trap_if(load_0_1 < 0 or load_0_1 >= 49152)
            load_1_1 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2_1] if 0 <= (o2_1) < 576 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 576
            remainder_2 = remainder_2 // 576
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            trap_if(load_0_2 < 0 or load_0_2 >= 49152)
            load_1_2 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2_2] if 0 <= (o2_2) < 576 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 576
            remainder_3 = remainder_3 // 576
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            trap_if(load_0_3 < 0 or load_0_3 >= 49152)
            load_1_3 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2_3] if 0 <= (o2_3) < 576 else cutlass.BFloat16(0))
            value_0 = load_1_0
            value_1 = load_1_1
            value_2 = load_1_2
            value_3 = load_1_3
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3] = value_3

    @cute.jit
    def _body_1(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[1]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o2_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o2_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o2_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o2_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            value_0 = (load_0_0).to(cutlass.Float32)
            value_1 = (load_0_1).to(cutlass.Float32)
            value_2 = (load_0_2).to(cutlass.Float32)
            value_3 = (load_0_3).to(cutlass.Float32)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3] = value_3

    @cute.jit
    def _body_2(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        # One CTA tile per reduced row: the reduction runs once and every thread
        # then evaluates the post-reduction expression for that row's outputs.
        warp = thread // 32
        lane = thread % 32
        warps = 8
        output = tensor_bindings[2]
        for tile in cutlass.range(block, 1, workers):
            row_index = (cutlass.Int64(tile) * 1)
            if row_index < 1:
                o1 = (row_index // 1) % 1
                o0 = (row_index // 1) % 1
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for reduction_index in cutlass.range(thread, 576, 256, unroll_full=True):
                    r0 = reduction_index
                    accumulator = (accumulator + cute.pow(((tensor_bindings[0][o0, o1, r0] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (r0) < 576 else cutlass.BFloat16(0))).to(cutlass.Float32), (cutlass.Int64(2)).to(cutlass.Float32)))
                warp_partial = cute.arch.warp_reduction_sum(accumulator)
                if lane == 0:
                    scratch[ warp] = warp_partial
                cute.arch.sync_threads()
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for warp_index in cutlass.range(warps):
                    accumulator = accumulator + scratch[ warp_index]
                post_index_0 = thread + 0
                o2_0 = (post_index_0 // 1) % 576
                post_load_0_0 = (tensor_bindings[0][o0, o1, o2_0] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_0) < 576 else cutlass.BFloat16(0))
                post_load_1_0 = (tensor_bindings[1][o2_0] if 0 <= (o2_0) < 576 else cutlass.BFloat16(0))
                post_index_1 = thread + 256
                o2_1 = (post_index_1 // 1) % 576
                post_load_0_1 = (tensor_bindings[0][o0, o1, o2_1] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_1) < 576 else cutlass.BFloat16(0))
                post_load_1_1 = (tensor_bindings[1][o2_1] if 0 <= (o2_1) < 576 else cutlass.BFloat16(0))
                post_index_2 = thread + 512
                o2_2 = (post_index_2 // 1) % 576
                post_load_0_2 = (tensor_bindings[0][o0, o1, o2_2] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_2) < 576 else cutlass.BFloat16(0))
                post_load_1_2 = (tensor_bindings[1][o2_2] if 0 <= (o2_2) < 576 else cutlass.BFloat16(0))
                output[o0, o1, o2_0] = (((post_load_1_0).to(cutlass.Float32) * ((((post_load_0_0).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(576)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_1] = (((post_load_1_1).to(cutlass.Float32) * ((((post_load_0_1).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(576)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                if post_index_2 < 576:
                    output[o0, o1, o2_2] = (((post_load_1_2).to(cutlass.Float32) * ((((post_load_0_2).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(576)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                cute.arch.sync_threads()

    @cute.jit
    def _body_3(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[2]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 32
            remainder_0 = remainder_0 // 32
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_0 = (tensor_bindings[1][o2_0] if 0 <= (o2_0) < 32 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 32
            remainder_1 = remainder_1 // 32
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_1 = (tensor_bindings[1][o2_1] if 0 <= (o2_1) < 32 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 32
            remainder_2 = remainder_2 // 32
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_2 = (tensor_bindings[1][o2_2] if 0 <= (o2_2) < 32 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 32
            remainder_3 = remainder_3 // 32
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_3 = (tensor_bindings[1][o2_3] if 0 <= (o2_3) < 32 else cutlass.Float32(0))
            value_0 = (load_0_0 * load_1_0)
            value_1 = (load_0_1 * load_1_1)
            value_2 = (load_0_2 * load_1_2)
            value_3 = (load_0_3 * load_1_3)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3] = value_3

    @cute.jit
    def _body_5(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[1]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 32
            remainder_0 = remainder_0 // 32
            o2_0 = remainder_0 % 2
            remainder_0 = remainder_0 // 2
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 2 and 0 <= (o3_0) < 32 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 32
            remainder_1 = remainder_1 // 32
            o2_1 = remainder_1 % 2
            remainder_1 = remainder_1 // 2
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 2 and 0 <= (o3_1) < 32 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 32
            remainder_2 = remainder_2 // 32
            o2_2 = remainder_2 % 2
            remainder_2 = remainder_2 // 2
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 2 and 0 <= (o3_2) < 32 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 32
            remainder_3 = remainder_3 // 32
            o2_3 = remainder_3 % 2
            remainder_3 = remainder_3 // 2
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 2 and 0 <= (o3_3) < 32 else cutlass.Float32(0))
            value_0 = load_0_0
            value_1 = load_0_1
            value_2 = load_0_2
            value_3 = load_0_3
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_7(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[1]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 64
            remainder_0 = remainder_0 // 64
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 64 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 64 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 64 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 64 else cutlass.Float32(0))
            value_0 = ((cute.cos(load_0_0) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
            value_1 = ((cute.cos(load_0_1) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
            value_2 = ((cute.cos(load_0_2) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
            value_3 = ((cute.cos(load_0_3) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3] = value_3

    @cute.jit
    def _body_8(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[1]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 64
            remainder_0 = remainder_0 // 64
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 64 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 64 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 64 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 64 else cutlass.Float32(0))
            value_0 = ((cute.sin(load_0_0) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
            value_1 = ((cute.sin(load_0_1) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
            value_2 = ((cute.sin(load_0_2) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
            value_3 = ((cute.sin(load_0_3) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3] = value_3

    @cute.jit
    def _body_9(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        floats, counters, float_base, counter_base = epilogue
        current_length = tensor_bindings[5][0].to(cutlass.Int32)
        trap_if(current_length < 0 or current_length + cutlass.Int32(1) > cutlass.Int32(128))
        keys = current_length + cutlass.Int32(1)
        in_use = (keys + cutlass.Int32(63)) // cutlass.Int32(64)
        if in_use > cutlass.Int32(6):
            in_use = cutlass.Int32(6)
        chunk_keys = (keys + in_use - cutlass.Int32(1)) // in_use
        warp = thread // 32
        lane = thread % 32
        # Only the chunks in use are items, row-major so a row's items are
        # consecutive (see the deadlock note above).
        for item in cutlass.range(block, cutlass.Int32(9) * in_use, workers):
            bh = item // in_use
            chunk = item % in_use
            if True:
                b = bh // 9
                h = bh % 9
                kv_head = h // 3
                start = chunk * chunk_keys
                stop = start + chunk_keys
                if stop > keys:
                    stop = keys
                scores = float_base + bh * 128
                stats = float_base + 1152 + bh * 12
                # Cached keys come from the cache; the new key at current_length
                # (in the last chunk) from new_k and new_v.
                cached_stop = stop
                if cached_stop > current_length:
                    cached_stop = current_length
                # Every chunk in use runs the whole exchange, even with no keys.
                if True:
                    sequence = cutlass.Int32(0)
                    for d in cutlass.range(thread, 64, 256):
                        query_load_0 = (tensor_bindings[0][b, sequence, h, d] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (h) < 9 and 0 <= (d) < 64 else cutlass.BFloat16(0))
                        query_load_1 = (tensor_bindings[6][b, sequence, h, d] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (h) < 9 and 0 <= (d) < 64 else cutlass.BFloat16(0))
                        query_load_2 = (tensor_bindings[0][b, sequence, h, (32 + d)] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (h) < 9 and 0 <= ((32 + d)) < 64 else cutlass.BFloat16(0))
                        query_load_3 = (tensor_bindings[7][b, sequence, h, d] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (h) < 9 and 0 <= (d) < 64 else cutlass.BFloat16(0))
                        query_load_4 = (tensor_bindings[7][b, sequence, h, (-32 + d)] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (h) < 9 and 0 <= ((-32 + d)) < 64 else cutlass.BFloat16(0))
                        query_load_5 = (tensor_bindings[0][b, sequence, h, (-32 + d)] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (h) < 9 and 0 <= ((-32 + d)) < 64 else cutlass.BFloat16(0))
                        query_load_6 = (tensor_bindings[6][b, sequence, h, (-32 + d)] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (h) < 9 and 0 <= ((-32 + d)) < 64 else cutlass.BFloat16(0))
                        scratch[648 + d] = (((((((((query_load_0).to(cutlass.Float32) * (query_load_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) - ((((query_load_2).to(cutlass.Float32) * (query_load_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((d < cutlass.Int64(32))) else (((((((((query_load_0).to(cutlass.Float32) * (query_load_4).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + ((((query_load_5).to(cutlass.Float32) * (query_load_6).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((d < cutlass.Int64(64))) else (query_load_0))))).to(cutlass.Float32)
                    for d in cutlass.range(thread, 64, 256):
                        key_load_0 = (tensor_bindings[1][b, sequence, kv_head, d] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (kv_head) < 3 and 0 <= (d) < 64 else cutlass.BFloat16(0))
                        key_load_1 = (tensor_bindings[8][b, sequence, kv_head, d] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (kv_head) < 3 and 0 <= (d) < 64 else cutlass.BFloat16(0))
                        key_load_2 = (tensor_bindings[1][b, sequence, kv_head, (32 + d)] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (kv_head) < 3 and 0 <= ((32 + d)) < 64 else cutlass.BFloat16(0))
                        key_load_3 = (tensor_bindings[9][b, sequence, kv_head, d] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (kv_head) < 3 and 0 <= (d) < 64 else cutlass.BFloat16(0))
                        key_load_4 = (tensor_bindings[9][b, sequence, kv_head, (-32 + d)] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (kv_head) < 3 and 0 <= ((-32 + d)) < 64 else cutlass.BFloat16(0))
                        key_load_5 = (tensor_bindings[1][b, sequence, kv_head, (-32 + d)] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (kv_head) < 3 and 0 <= ((-32 + d)) < 64 else cutlass.BFloat16(0))
                        key_load_6 = (tensor_bindings[8][b, sequence, kv_head, (-32 + d)] if 0 <= (b) < 1 and 0 <= (sequence) < 1 and 0 <= (kv_head) < 3 and 0 <= ((-32 + d)) < 64 else cutlass.BFloat16(0))
                        scratch[712 + d] = (((((((((key_load_0).to(cutlass.Float32) * (key_load_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) - ((((key_load_2).to(cutlass.Float32) * (key_load_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((d < cutlass.Int64(32))) else (((((((((key_load_0).to(cutlass.Float32) * (key_load_4).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + ((((key_load_5).to(cutlass.Float32) * (key_load_6).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((d < cutlass.Int64(64))) else (key_load_0))))).to(cutlass.Float32)
                    cute.arch.sync_threads()
                    query = cute.make_tensor(scratch.iterator + 648, cute.make_layout(64)).load()
                    # Scores: one key per thread, whole rows per load.
                    maximum = cutlass.Float32(-1.0e30)
                    for key in cutlass.range(start + thread, cached_stop, 256):
                        row = tensor_bindings[3][(b, key, kv_head, None)].load().to(cutlass.Float32)
                        score = cutlass.Float32((query * row).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0))
                        score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                        score = score * cutlass.Float32(0.125)
                        scratch[520 + key] = score
                        maximum = cute.arch.fmax(maximum, score, nan=True)
                    if stop == keys:
                        if thread == 0:
                            row = cute.make_tensor(scratch.iterator + 712, cute.make_layout(64)).load()
                            score = cutlass.Float32((query * row).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0))
                            score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                            score = score * cutlass.Float32(0.125)
                            scratch[520 + current_length] = score
                            maximum = cute.arch.fmax(maximum, score, nan=True)
                    maximum = cute.arch.warp_reduction_max(maximum)
                    if lane == 0:
                        scratch[warp] = maximum
                    cute.arch.sync_threads()
                    if thread == 0:
                        chunk_maximum = scratch[0]
                        for other in cutlass.range_constexpr(1, 8):
                            chunk_maximum = cute.arch.fmax(chunk_maximum, scratch[other], nan=True)
                        if in_use > cutlass.Int32(1):
                            cute.arch.store(floats.iterator + stats + chunk * 2, chunk_maximum, cop="cg")
                        scratch[0] = chunk_maximum
                    if in_use > cutlass.Int32(1):
                        cute.arch.fence_acq_rel_gpu()
                        if thread == 0:
                            cute.arch.atomic_add(
                                counters.iterator + counter_base + bh * 3 + 0,
                                cutlass.Int32(1),
                                sem="release",
                                scope="gpu",
                            )
                        cute.arch.sync_threads()
                        if thread == 0:
                            seen = cutlass.Int32(0)
                            while seen < in_use:
                                seen = cute.arch.atomic_cas(
                                    counters.iterator + counter_base + bh * 3 + 0,
                                    cmp=in_use,
                                    val=in_use,
                                    sem="acquire",
                                    scope="gpu",
                                )
                        cute.arch.sync_threads()
                        cute.arch.fence_acq_rel_gpu()
                        if thread == 0:
                            row_maximum = cute.arch.load(floats.iterator + stats, cutlass.Float32, cop="cg")
                            for other in cutlass.range(1, in_use):
                                row_maximum = cute.arch.fmax(
                                    row_maximum,
                                    cute.arch.load(floats.iterator + stats + other * 2, cutlass.Float32, cop="cg"),
                                    nan=True,
                                )
                            scratch[0] = row_maximum
                    cute.arch.sync_threads()
                    row_maximum = scratch[0]
                    # Exponentials of this chunk's scores and their sum.
                    total = cutlass.Float32(0.0)
                    for key in cutlass.range(start + thread, stop, 256):
                        probability = cute.exp(scratch[520 + key] - row_maximum)
                        scratch[520 + key] = probability
                        total = total + probability
                    total = cute.arch.warp_reduction_sum(total)
                    cute.arch.sync_threads()
                    if lane == 0:
                        scratch[warp] = total
                    cute.arch.sync_threads()
                    if thread == 0:
                        chunk_total = scratch[0]
                        for other in cutlass.range_constexpr(1, 8):
                            chunk_total = chunk_total + scratch[other]
                        if in_use > cutlass.Int32(1):
                            cute.arch.store(floats.iterator + stats + chunk * 2 + 1, chunk_total, cop="cg")
                        scratch[0] = chunk_total
                    if in_use > cutlass.Int32(1):
                        cute.arch.fence_acq_rel_gpu()
                        if thread == 0:
                            cute.arch.atomic_add(
                                counters.iterator + counter_base + bh * 3 + 1,
                                cutlass.Int32(1),
                                sem="release",
                                scope="gpu",
                            )
                        cute.arch.sync_threads()
                        if thread == 0:
                            seen = cutlass.Int32(0)
                            while seen < in_use:
                                seen = cute.arch.atomic_cas(
                                    counters.iterator + counter_base + bh * 3 + 1,
                                    cmp=in_use,
                                    val=in_use,
                                    sem="acquire",
                                    scope="gpu",
                                )
                        cute.arch.sync_threads()
                        cute.arch.fence_acq_rel_gpu()
                        if thread == 0:
                            row_total = cutlass.Float32(0.0)
                            for other in cutlass.range(0, in_use):
                                row_total = row_total + cute.arch.load(
                                    floats.iterator + stats + other * 2 + 1, cutlass.Float32, cop="cg"
                                )
                            scratch[0] = row_total
                    cute.arch.sync_threads()
                    denominator = scratch[0]
                    if denominator <= cutlass.Float32(0.0):
                        denominator = cutlass.Float32(1.0)
                    cute.arch.sync_threads()
                    value_0 = cutlass.Float32(0.0)
                    value_1 = cutlass.Float32(0.0)
                    for base in cutlass.range(start + warp, cached_stop, 64):
                        key_0 = base + 0
                        probability_0 = cutlass.Float32(0.0)
                        if key_0 < cached_stop:
                            probability_0 = (
                                scratch[520 + key_0] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_0 = cached_stop - cutlass.Int32(1)
                        row_0_0 = tensor_bindings[4][b, key_0, kv_head, lane + 0].to(cutlass.Float32)
                        row_0_1 = tensor_bindings[4][b, key_0, kv_head, lane + 32].to(cutlass.Float32)
                        key_1 = base + 8
                        probability_1 = cutlass.Float32(0.0)
                        if key_1 < cached_stop:
                            probability_1 = (
                                scratch[520 + key_1] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_1 = cached_stop - cutlass.Int32(1)
                        row_1_0 = tensor_bindings[4][b, key_1, kv_head, lane + 0].to(cutlass.Float32)
                        row_1_1 = tensor_bindings[4][b, key_1, kv_head, lane + 32].to(cutlass.Float32)
                        key_2 = base + 16
                        probability_2 = cutlass.Float32(0.0)
                        if key_2 < cached_stop:
                            probability_2 = (
                                scratch[520 + key_2] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_2 = cached_stop - cutlass.Int32(1)
                        row_2_0 = tensor_bindings[4][b, key_2, kv_head, lane + 0].to(cutlass.Float32)
                        row_2_1 = tensor_bindings[4][b, key_2, kv_head, lane + 32].to(cutlass.Float32)
                        key_3 = base + 24
                        probability_3 = cutlass.Float32(0.0)
                        if key_3 < cached_stop:
                            probability_3 = (
                                scratch[520 + key_3] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_3 = cached_stop - cutlass.Int32(1)
                        row_3_0 = tensor_bindings[4][b, key_3, kv_head, lane + 0].to(cutlass.Float32)
                        row_3_1 = tensor_bindings[4][b, key_3, kv_head, lane + 32].to(cutlass.Float32)
                        key_4 = base + 32
                        probability_4 = cutlass.Float32(0.0)
                        if key_4 < cached_stop:
                            probability_4 = (
                                scratch[520 + key_4] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_4 = cached_stop - cutlass.Int32(1)
                        row_4_0 = tensor_bindings[4][b, key_4, kv_head, lane + 0].to(cutlass.Float32)
                        row_4_1 = tensor_bindings[4][b, key_4, kv_head, lane + 32].to(cutlass.Float32)
                        key_5 = base + 40
                        probability_5 = cutlass.Float32(0.0)
                        if key_5 < cached_stop:
                            probability_5 = (
                                scratch[520 + key_5] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_5 = cached_stop - cutlass.Int32(1)
                        row_5_0 = tensor_bindings[4][b, key_5, kv_head, lane + 0].to(cutlass.Float32)
                        row_5_1 = tensor_bindings[4][b, key_5, kv_head, lane + 32].to(cutlass.Float32)
                        key_6 = base + 48
                        probability_6 = cutlass.Float32(0.0)
                        if key_6 < cached_stop:
                            probability_6 = (
                                scratch[520 + key_6] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_6 = cached_stop - cutlass.Int32(1)
                        row_6_0 = tensor_bindings[4][b, key_6, kv_head, lane + 0].to(cutlass.Float32)
                        row_6_1 = tensor_bindings[4][b, key_6, kv_head, lane + 32].to(cutlass.Float32)
                        key_7 = base + 56
                        probability_7 = cutlass.Float32(0.0)
                        if key_7 < cached_stop:
                            probability_7 = (
                                scratch[520 + key_7] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_7 = cached_stop - cutlass.Int32(1)
                        row_7_0 = tensor_bindings[4][b, key_7, kv_head, lane + 0].to(cutlass.Float32)
                        row_7_1 = tensor_bindings[4][b, key_7, kv_head, lane + 32].to(cutlass.Float32)
                        value_0 = value_0 + probability_0 * row_0_0
                        value_1 = value_1 + probability_0 * row_0_1
                        value_0 = value_0 + probability_1 * row_1_0
                        value_1 = value_1 + probability_1 * row_1_1
                        value_0 = value_0 + probability_2 * row_2_0
                        value_1 = value_1 + probability_2 * row_2_1
                        value_0 = value_0 + probability_3 * row_3_0
                        value_1 = value_1 + probability_3 * row_3_1
                        value_0 = value_0 + probability_4 * row_4_0
                        value_1 = value_1 + probability_4 * row_4_1
                        value_0 = value_0 + probability_5 * row_5_0
                        value_1 = value_1 + probability_5 * row_5_1
                        value_0 = value_0 + probability_6 * row_6_0
                        value_1 = value_1 + probability_6 * row_6_1
                        value_0 = value_0 + probability_7 * row_7_0
                        value_1 = value_1 + probability_7 * row_7_1
                    if stop == keys:
                        if warp == 0:
                            probability = (
                                scratch[520 + current_length] / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                            value_0 = value_0 + probability * tensor_bindings[2][b, 0, kv_head, lane + 0].to(cutlass.Float32)
                            value_1 = value_1 + probability * tensor_bindings[2][b, 0, kv_head, lane + 32].to(cutlass.Float32)
                    scratch[8 + warp * 64 + lane + 0] = value_0
                    scratch[8 + warp * 64 + lane + 32] = value_1
                    cute.arch.sync_threads()
                    if in_use == cutlass.Int32(1):
                        for d in cutlass.range(thread, 64, 256):
                            value = scratch[8 + d]
                            for other in cutlass.range_constexpr(1, 8):
                                value = value + scratch[8 + other * 64 + d]
                            tensor_bindings[10][bh, 0, d] = value.to(cutlass.BFloat16)
                    else:
                        partials = float_base + 1260 + bh * 384
                        for d in cutlass.range(thread, 64, 256):
                            value = scratch[8 + d]
                            for other in cutlass.range_constexpr(1, 8):
                                value = value + scratch[8 + other * 64 + d]
                            cute.arch.store(
                                floats.iterator + partials + chunk * 64 + d, value, cop="cg"
                            )
                        cute.arch.fence_acq_rel_gpu()
                        cute.arch.sync_threads()
                        if thread == 0:
                            arrived = cute.arch.atomic_add(
                                counters.iterator + counter_base + bh * 3 + 2,
                                cutlass.Int32(1),
                                sem="acq_rel",
                                scope="gpu",
                            )
                            scratch[0] = cutlass.Float32(0.0)
                            if arrived == in_use - cutlass.Int32(1):
                                scratch[0] = cutlass.Float32(1.0)
                        cute.arch.sync_threads()
                        if scratch[0] != cutlass.Float32(0.0):
                            cute.arch.fence_acq_rel_gpu()
                            for d in cutlass.range(thread, 64, 256):
                                value = cutlass.Float32(0.0)
                                for other in cutlass.range(0, in_use):
                                    value = value + cute.arch.load(
                                        floats.iterator + partials + other * 64 + d,
                                        cutlass.Float32,
                                        cop="cg",
                                    )
                                tensor_bindings[10][bh, 0, d] = value.to(cutlass.BFloat16)
                            if thread == 0:
                                # Every chunk has passed both waits: reset for the next phase.
                                for counter in cutlass.range_constexpr(3):
                                    cute.arch.atomic_exch(
                                        counters.iterator + counter_base + bh * 3 + counter,
                                        cutlass.Int32(0),
                                        sem="relaxed",
                                        scope="gpu",
                                    )
                    cute.arch.sync_threads()
                # Keys before current_length come from the cache and the new key
                # from new_k, so appending here cannot race with readers.
                if chunk == 0:
                    if h % 3 == 0:
                        if warp == 0:
                            for d in cutlass.range(lane, 64, 32):
                                tensor_bindings[3][b, current_length, kv_head, d] = scratch[712 + d].to(cutlass.BFloat16)
                                tensor_bindings[4][b, current_length, kv_head, d] = tensor_bindings[2][b, 0, kv_head, d]
                cute.arch.sync_threads()


    # Publish this call's boundary addresses before any phase reads them.
    if _block == 0:
        _dynamic = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int64,
                dynamic_addresses + cutlass.Int64(_call % 64) * 9320,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            cute.make_layout(1165),
        )
        trap_if(_dynamic[0] != cutlass.Int64(_call))
        for _position in cutlass.range(_thread, 1164, 256, unroll=1):
            addresses[dynamic_slot_table[_position]] = _dynamic[_position + 1]

    counting_grid_barrier(ready, workers, _ordinal)
    if _block == 0:
        if _thread == 0:
            ready[workers + 2] = _call + cutlass.Int32(1)
    _warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if _warp == 0:
        for gemm in cutlass.range(_block, 181, workers, unroll=1):
            gemm_variant = cute.arch.make_warp_uniform(gemm_table[gemm, 0])
            if gemm_variant == 0:
                build_table_descriptors(
                    gemm_values[0],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (576, 1, 576),
                    True,
                    1,
                )
            if gemm_variant == 1:
                build_table_descriptors(
                    gemm_values[1],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (192, 1, 576),
                    True,
                    1,
                )
            if gemm_variant == 2:
                build_table_descriptors(
                    gemm_values[2],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (1536, 1, 576),
                    True,
                    1,
                    addresses[gemm_table[gemm, 9] - 1],
                )
            if gemm_variant == 3:
                build_table_descriptors(
                    gemm_values[3],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (576, 1, 1536),
                    True,
                    1,
                )
            if gemm_variant == 4:
                build_table_descriptors(
                    gemm_values[4],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (49152, 1, 576),
                    True,
                    1,
                )
        cute.nvgpu.cpasync.fence_tma_desc_release()
    if _thread == 0:
        _gemm_scratch.stream_tag[0] = cutlass.Int32(0)
    counting_grid_barrier(ready, workers, _ordinal + cutlass.Int32(1))

    for phase in cutlass.range(278, unroll=1):
        if _thread == 0:
            if phase + 1 < 278:
                prefetch_l1(phase_table.iterator + (phase + 1) * 21)
                prefetch_l1(addresses.iterator + 1345 + (phase + 1) * 11)
        # Table entries are the same on every thread; saying so lets ptxas
        # treat the dispatch branches around the GEMM bodies as warp-uniform.
        variant = cute.arch.make_warp_uniform(phase_table[phase, 0])
        _rotation = cute.arch.make_warp_uniform(phase_table[phase, 2])
        _units = cute.arch.make_warp_uniform(phase_table[phase, 19])
        _rotated = (_block + workers - _rotation) % workers
        _workers = workers
        if _rotation < 0:
            # A recompute phase: this CTA runs every tile itself.
            _rotated = cutlass.Int32(0)
            _workers = cutlass.Int32(1)
        if _rotated < _units:
            if variant == 0:
                _body_0(
                    work_iterators[0],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1), stride=(1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((49152, 576), stride=(576, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2),),
                    None,
                    None,
                )
            if variant == 1:
                _body_1(
                    work_iterators[1],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1, 1), stride=(1, 1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((1, 1, 1), stride=(1, 1, 1)), 4),),
                    None,
                    None,
                )
            if variant == 2:
                _body_2(
                    work_iterators[2],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((576,), stride=(1,)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2),),
                    _scratch_2,
                    None,
                )
            if variant == 3:
                _body_3(
                    work_iterators[3],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 1), stride=(1, 1, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((32,), stride=(1,)), 4), _slot(addresses, phase_table, phase, 2, cutlass.Float32, cute.make_layout((1, 1, 32), stride=(32, 32, 1)), 4),),
                    None,
                    None,
                )
            if variant == 4:
                _gemm_0.run_table(
                    gemm_values[0],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                    epilogue_state=lambda: (
                        gemm_table[phase_table[phase, 6], 10],
                        (
                            (_gemm_slot(addresses, gemm_table, phase_table[phase, 6], 11, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2), ),
                        ),
                    ),
                    stream_tag=phase_table[phase, 6] + 1,
                )
            if variant == 5:
                _body_5(
                    work_iterators[4],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 2, 32), stride=(32, 32, 0, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((1, 1, 2, 32), stride=(64, 64, 32, 1)), 4),),
                    None,
                    None,
                )
            if variant == 6:
                _gemm_1.run_table(
                    gemm_values[1],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                    stream_tag=phase_table[phase, 6] + 1,
                )
            if variant == 7:
                _body_7(
                    work_iterators[5],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 2),),
                    None,
                    None,
                )
            if variant == 8:
                _body_8(
                    work_iterators[6],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 2),),
                    None,
                    None,
                )
            if variant == 9:
                _body_9(
                    work_iterators[7],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 9, 64), stride=(576, 576, 64, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(192, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(192, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 128, 3, 64), stride=(24576, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 4, cutlass.BFloat16, cute.make_layout((1, 128, 3, 64), stride=(24576, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 5, cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4), _slot(addresses, phase_table, phase, 6, cutlass.BFloat16, cute.make_layout((1, 1, 9, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 7, cutlass.BFloat16, cute.make_layout((1, 1, 9, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 8, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 9, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 10, cutlass.BFloat16, cute.make_layout((9, 1, 64), stride=(64, 64, 1)), 2),),
                    _scratch_10,
                    (phase_floats, phase_counters, phase_table[phase, 4], phase_table[phase, 5]),
                )
            if variant == 10:
                _gemm_2.run_table(
                    gemm_values[2],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                    epilogue_state=lambda: (
                        gemm_table[phase_table[phase, 6], 10],
                        (
                            (),
                        ),
                    ),
                    stream_tag=phase_table[phase, 6] + 1,
                )
            if variant == 11:
                _gemm_3.run_table(
                    gemm_values[3],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                    epilogue_state=lambda: (
                        gemm_table[phase_table[phase, 6], 10],
                        (
                            (_gemm_slot(addresses, gemm_table, phase_table[phase, 6], 11, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2), ),
                        ),
                    ),
                    stream_tag=phase_table[phase, 6] + 1,
                )
            if variant == 12:
                _gemm_4.run_table(
                    gemm_values[4],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                    stream_tag=phase_table[phase, 6] + 1,
                )
        if phase_table[phase, 2] < 0:
            # Later TMA reads in this interval see the generic-proxy writes.
            cute.arch.fence_proxy("async.global")
        if phase_table[phase, 1] != 0:
            counting_grid_arrive(ready, workers)
            _stream = cute.arch.make_warp_uniform(phase_table[phase, 20])
            if _stream != 0:
                if _warp == 0:
                    _target = cute.arch.make_warp_uniform(stream_table[_stream, 3 * _block])
                    if _target != 0:
                        _target_variant = cute.arch.make_warp_uniform(
                            stream_table[_stream, 3 * _block + 1]
                        )
                        _target_block = cute.arch.make_warp_uniform(
                            stream_table[_stream, 3 * _block + 2]
                        )
                        if _target_variant == 0:
                            _gemm_0.preload_table(
                                gemm_values[0],
                                _gemm_scratch,
                                table_descriptors(tma_descriptors, _target - 1),
                                _target_block,
                                _target,
                            )
                        if _target_variant == 1:
                            _gemm_1.preload_table(
                                gemm_values[1],
                                _gemm_scratch,
                                table_descriptors(tma_descriptors, _target - 1),
                                _target_block,
                                _target,
                            )
                        if _target_variant == 2:
                            _gemm_2.preload_table(
                                gemm_values[2],
                                _gemm_scratch,
                                table_descriptors(tma_descriptors, _target - 1),
                                _target_block,
                                _target,
                            )
                        if _target_variant == 3:
                            _gemm_3.preload_table(
                                gemm_values[3],
                                _gemm_scratch,
                                table_descriptors(tma_descriptors, _target - 1),
                                _target_block,
                                _target,
                            )
                        if _target_variant == 4:
                            _gemm_4.preload_table(
                                gemm_values[4],
                                _gemm_scratch,
                                table_descriptors(tma_descriptors, _target - 1),
                                _target_block,
                                _target,
                            )
            counting_grid_wait(ready, workers, _ordinal + phase_table[phase, 1])
            # Every reader of this interval's KV valid length has finished.
            if phase_table[phase, 3] != 0:
                if _block == 0:
                    if _thread == 0:
                        _length = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Int32,
                                addresses[phase_table[phase, 3] - 1],
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(1),
                        )
                        _length[0] = _length[0] + cutlass.Int32(1)
        else:
            # Independent phases share a barrier interval; the CTA still
            # synchronizes before the next body reuses its shared scratch.
            cute.arch.sync_threads()


@cute.jit
def launch(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    work_iterators: tuple[cute.Tensor, ...],
    dynamic_addresses: cutlass.Int64,
    dynamic_slot_table: cute.Tensor,
    phase_floats: cute.Tensor,
    phase_counters: cute.Tensor,
    gemm_table: cute.Tensor,
    stream_table: cute.Tensor,
    tma_descriptors: cute.Tensor,
    gemm_tensors: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...],
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    gemm_values = (
        _gemm_0.prepare(
            gemm_tensors[0][0],
            gemm_tensors[0][1],
            gemm_tensors[0][2],
            max_active_clusters,
        ),
        _gemm_1.prepare(
            gemm_tensors[1][0],
            gemm_tensors[1][1],
            gemm_tensors[1][2],
            max_active_clusters,
        ),
        _gemm_2.prepare(
            gemm_tensors[2][0],
            gemm_tensors[2][1],
            gemm_tensors[2][2],
            max_active_clusters,
        ),
        _gemm_3.prepare(
            gemm_tensors[3][0],
            gemm_tensors[3][1],
            gemm_tensors[3][2],
            max_active_clusters,
        ),
        _gemm_4.prepare(
            gemm_tensors[4][0],
            gemm_tensors[4][1],
            gemm_tensors[4][2],
            max_active_clusters,
        ),
    )
    megabake_kernel(
        ready,
        workers,
        addresses,
        phase_table,
        work_iterators,
        dynamic_addresses,
        dynamic_slot_table,
        phase_floats,
        phase_counters,
        gemm_table,
        stream_table,
        tma_descriptors,
        gemm_values,
    ).launch(
        grid=[1, 1, workers],
        block=[256, 1, 1],
        cooperative=True,
        stream=stream,
    )

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cute.tensor import _Tensor

from megabake.runtime import counting_grid_barrier
from megabake.bodies.gemm import (
    HopperWgmmaGemmPersistentKernel,
    build_table_descriptors,
    table_descriptors,
)

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
)
_gemm_2 = HopperWgmmaGemmPersistentKernel(
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
    address = addresses[phase_table[phase, 6 + column]]
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
class _Phase25Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase127Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 2048],
        16
    ]


@cute.union
class _WorkerScratch:
    phase_25: _Phase25Scratch
    phase_127: _Phase127Scratch

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
    tma_descriptors: cute.Tensor,
    gemm_values,
):
    _, _, _block = cute.arch.block_idx()
    _thread, _, _ = cute.arch.thread_idx()
    # Every CTA reads the call counter before the first barrier; block 0
    # advances it after that barrier, for the next launch on this stream.
    _call = ready[workers + 2]
    _ordinal = _call * cutlass.Int32(173)
    _worker_scratch = cutlass.utils.SmemAllocator().allocate(_WorkerScratch)
    _scratch_25 = _worker_scratch.phase_25.body.get_tensor(cute.make_layout(8))
    _scratch_127 = _worker_scratch.phase_127.body.get_tensor(cute.make_layout(2048))
    if _thread == 0:
        _worker_scratch.phase_25.barrier_0[0] = 0
    cute.arch.sync_threads()

    _gemm_scratch = cutlass.utils.SmemAllocator().allocate(_gemm_0.shared_storage)

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 2048
            remainder_0 = remainder_0 // 2048
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            if load_0_0 < 0:
                llvm.intr_trap()
            if load_0_0 >= 49152:
                llvm.intr_trap()
            load_1_0 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2_0] if 0 <= (o2_0) < 2048 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 2048
            remainder_1 = remainder_1 // 2048
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            if load_0_1 < 0:
                llvm.intr_trap()
            if load_0_1 >= 49152:
                llvm.intr_trap()
            load_1_1 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2_1] if 0 <= (o2_1) < 2048 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 2048
            remainder_2 = remainder_2 // 2048
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            if load_0_2 < 0:
                llvm.intr_trap()
            if load_0_2 >= 49152:
                llvm.intr_trap()
            load_1_2 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2_2] if 0 <= (o2_2) < 2048 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 2048
            remainder_3 = remainder_3 // 2048
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            if load_0_3 < 0:
                llvm.intr_trap()
            if load_0_3 >= 49152:
                llvm.intr_trap()
            load_1_3 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2_3] if 0 <= (o2_3) < 2048 else cutlass.BFloat16(0))
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            row_index = work_iterator[tile]
            if row_index < 1:
                o1 = (row_index // 1) % 1
                o0 = (row_index // 1) % 1
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for reduction_index in cutlass.range(thread, 2048, 256, unroll_full=True):
                    r0 = reduction_index
                    accumulator = (accumulator + cute.pow(((tensor_bindings[0][o0, o1, r0] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (r0) < 2048 else cutlass.BFloat16(0))).to(cutlass.Float32), (cutlass.Int64(2)).to(cutlass.Float32)))
                warp_partial = cute.arch.warp_reduction_sum(accumulator)
                if lane == 0:
                    scratch[ warp] = warp_partial
                cute.arch.sync_threads()
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for warp_index in cutlass.range(warps):
                    accumulator = accumulator + scratch[ warp_index]
                post_index_0 = thread + 0
                o2_0 = (post_index_0 // 1) % 2048
                post_load_0_0 = (tensor_bindings[0][o0, o1, o2_0] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_0) < 2048 else cutlass.BFloat16(0))
                post_load_1_0 = (tensor_bindings[1][o2_0] if 0 <= (o2_0) < 2048 else cutlass.BFloat16(0))
                post_index_1 = thread + 256
                o2_1 = (post_index_1 // 1) % 2048
                post_load_0_1 = (tensor_bindings[0][o0, o1, o2_1] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_1) < 2048 else cutlass.BFloat16(0))
                post_load_1_1 = (tensor_bindings[1][o2_1] if 0 <= (o2_1) < 2048 else cutlass.BFloat16(0))
                post_index_2 = thread + 512
                o2_2 = (post_index_2 // 1) % 2048
                post_load_0_2 = (tensor_bindings[0][o0, o1, o2_2] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_2) < 2048 else cutlass.BFloat16(0))
                post_load_1_2 = (tensor_bindings[1][o2_2] if 0 <= (o2_2) < 2048 else cutlass.BFloat16(0))
                post_index_3 = thread + 768
                o2_3 = (post_index_3 // 1) % 2048
                post_load_0_3 = (tensor_bindings[0][o0, o1, o2_3] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_3) < 2048 else cutlass.BFloat16(0))
                post_load_1_3 = (tensor_bindings[1][o2_3] if 0 <= (o2_3) < 2048 else cutlass.BFloat16(0))
                post_index_4 = thread + 1024
                o2_4 = (post_index_4 // 1) % 2048
                post_load_0_4 = (tensor_bindings[0][o0, o1, o2_4] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_4) < 2048 else cutlass.BFloat16(0))
                post_load_1_4 = (tensor_bindings[1][o2_4] if 0 <= (o2_4) < 2048 else cutlass.BFloat16(0))
                post_index_5 = thread + 1280
                o2_5 = (post_index_5 // 1) % 2048
                post_load_0_5 = (tensor_bindings[0][o0, o1, o2_5] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_5) < 2048 else cutlass.BFloat16(0))
                post_load_1_5 = (tensor_bindings[1][o2_5] if 0 <= (o2_5) < 2048 else cutlass.BFloat16(0))
                post_index_6 = thread + 1536
                o2_6 = (post_index_6 // 1) % 2048
                post_load_0_6 = (tensor_bindings[0][o0, o1, o2_6] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_6) < 2048 else cutlass.BFloat16(0))
                post_load_1_6 = (tensor_bindings[1][o2_6] if 0 <= (o2_6) < 2048 else cutlass.BFloat16(0))
                post_index_7 = thread + 1792
                o2_7 = (post_index_7 // 1) % 2048
                post_load_0_7 = (tensor_bindings[0][o0, o1, o2_7] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2_7) < 2048 else cutlass.BFloat16(0))
                post_load_1_7 = (tensor_bindings[1][o2_7] if 0 <= (o2_7) < 2048 else cutlass.BFloat16(0))
                output[o0, o1, o2_0] = (((post_load_1_0).to(cutlass.Float32) * ((((post_load_0_0).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(2048)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_1] = (((post_load_1_1).to(cutlass.Float32) * ((((post_load_0_1).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(2048)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_2] = (((post_load_1_2).to(cutlass.Float32) * ((((post_load_0_2).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(2048)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_3] = (((post_load_1_3).to(cutlass.Float32) * ((((post_load_0_3).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(2048)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_4] = (((post_load_1_4).to(cutlass.Float32) * ((((post_load_0_4).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(2048)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_5] = (((post_load_1_5).to(cutlass.Float32) * ((((post_load_0_5).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(2048)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_6] = (((post_load_1_6).to(cutlass.Float32) * ((((post_load_0_6).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(2048)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_7] = (((post_load_1_7).to(cutlass.Float32) * ((((post_load_0_7).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(2048)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
    def _body_6(
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
    def _body_8(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[3]
        output_count = cute.size(output)
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 64
            remainder_0 = remainder_0 // 64
            o2_0 = remainder_0 % 32
            remainder_0 = remainder_0 // 32
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][o0_0, o1_0, o2_0, o3_0] if 0 <= (o0_0) < 1 and 0 <= (o1_0) < 1 and 0 <= (o2_0) < 32 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[1][o0_0, o1_0, o2_0, o3_0] if 0 <= (o0_0) < 1 and 0 <= (o1_0) < 1 and 0 <= (o2_0) < 32 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            load_2_0 = (tensor_bindings[0][o0_0, o1_0, o2_0, (32 + o3_0)] if 0 <= (o0_0) < 1 and 0 <= (o1_0) < 1 and 0 <= (o2_0) < 32 and 0 <= ((32 + o3_0)) < 64 else cutlass.BFloat16(0))
            load_3_0 = (tensor_bindings[2][o0_0, o1_0, o2_0, o3_0] if 0 <= (o0_0) < 1 and 0 <= (o1_0) < 1 and 0 <= (o2_0) < 32 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            load_4_0 = (tensor_bindings[2][o0_0, o1_0, o2_0, (-32 + o3_0)] if 0 <= (o0_0) < 1 and 0 <= (o1_0) < 1 and 0 <= (o2_0) < 32 and 0 <= ((-32 + o3_0)) < 64 else cutlass.BFloat16(0))
            load_5_0 = (tensor_bindings[0][o0_0, o1_0, o2_0, (-32 + o3_0)] if 0 <= (o0_0) < 1 and 0 <= (o1_0) < 1 and 0 <= (o2_0) < 32 and 0 <= ((-32 + o3_0)) < 64 else cutlass.BFloat16(0))
            load_6_0 = (tensor_bindings[1][o0_0, o1_0, o2_0, (-32 + o3_0)] if 0 <= (o0_0) < 1 and 0 <= (o1_0) < 1 and 0 <= (o2_0) < 32 and 0 <= ((-32 + o3_0)) < 64 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o2_1 = remainder_1 % 32
            remainder_1 = remainder_1 // 32
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][o0_1, o1_1, o2_1, o3_1] if 0 <= (o0_1) < 1 and 0 <= (o1_1) < 1 and 0 <= (o2_1) < 32 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[1][o0_1, o1_1, o2_1, o3_1] if 0 <= (o0_1) < 1 and 0 <= (o1_1) < 1 and 0 <= (o2_1) < 32 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            load_2_1 = (tensor_bindings[0][o0_1, o1_1, o2_1, (32 + o3_1)] if 0 <= (o0_1) < 1 and 0 <= (o1_1) < 1 and 0 <= (o2_1) < 32 and 0 <= ((32 + o3_1)) < 64 else cutlass.BFloat16(0))
            load_3_1 = (tensor_bindings[2][o0_1, o1_1, o2_1, o3_1] if 0 <= (o0_1) < 1 and 0 <= (o1_1) < 1 and 0 <= (o2_1) < 32 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            load_4_1 = (tensor_bindings[2][o0_1, o1_1, o2_1, (-32 + o3_1)] if 0 <= (o0_1) < 1 and 0 <= (o1_1) < 1 and 0 <= (o2_1) < 32 and 0 <= ((-32 + o3_1)) < 64 else cutlass.BFloat16(0))
            load_5_1 = (tensor_bindings[0][o0_1, o1_1, o2_1, (-32 + o3_1)] if 0 <= (o0_1) < 1 and 0 <= (o1_1) < 1 and 0 <= (o2_1) < 32 and 0 <= ((-32 + o3_1)) < 64 else cutlass.BFloat16(0))
            load_6_1 = (tensor_bindings[1][o0_1, o1_1, o2_1, (-32 + o3_1)] if 0 <= (o0_1) < 1 and 0 <= (o1_1) < 1 and 0 <= (o2_1) < 32 and 0 <= ((-32 + o3_1)) < 64 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o2_2 = remainder_2 % 32
            remainder_2 = remainder_2 // 32
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][o0_2, o1_2, o2_2, o3_2] if 0 <= (o0_2) < 1 and 0 <= (o1_2) < 1 and 0 <= (o2_2) < 32 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[1][o0_2, o1_2, o2_2, o3_2] if 0 <= (o0_2) < 1 and 0 <= (o1_2) < 1 and 0 <= (o2_2) < 32 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            load_2_2 = (tensor_bindings[0][o0_2, o1_2, o2_2, (32 + o3_2)] if 0 <= (o0_2) < 1 and 0 <= (o1_2) < 1 and 0 <= (o2_2) < 32 and 0 <= ((32 + o3_2)) < 64 else cutlass.BFloat16(0))
            load_3_2 = (tensor_bindings[2][o0_2, o1_2, o2_2, o3_2] if 0 <= (o0_2) < 1 and 0 <= (o1_2) < 1 and 0 <= (o2_2) < 32 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            load_4_2 = (tensor_bindings[2][o0_2, o1_2, o2_2, (-32 + o3_2)] if 0 <= (o0_2) < 1 and 0 <= (o1_2) < 1 and 0 <= (o2_2) < 32 and 0 <= ((-32 + o3_2)) < 64 else cutlass.BFloat16(0))
            load_5_2 = (tensor_bindings[0][o0_2, o1_2, o2_2, (-32 + o3_2)] if 0 <= (o0_2) < 1 and 0 <= (o1_2) < 1 and 0 <= (o2_2) < 32 and 0 <= ((-32 + o3_2)) < 64 else cutlass.BFloat16(0))
            load_6_2 = (tensor_bindings[1][o0_2, o1_2, o2_2, (-32 + o3_2)] if 0 <= (o0_2) < 1 and 0 <= (o1_2) < 1 and 0 <= (o2_2) < 32 and 0 <= ((-32 + o3_2)) < 64 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o2_3 = remainder_3 % 32
            remainder_3 = remainder_3 // 32
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][o0_3, o1_3, o2_3, o3_3] if 0 <= (o0_3) < 1 and 0 <= (o1_3) < 1 and 0 <= (o2_3) < 32 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[1][o0_3, o1_3, o2_3, o3_3] if 0 <= (o0_3) < 1 and 0 <= (o1_3) < 1 and 0 <= (o2_3) < 32 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            load_2_3 = (tensor_bindings[0][o0_3, o1_3, o2_3, (32 + o3_3)] if 0 <= (o0_3) < 1 and 0 <= (o1_3) < 1 and 0 <= (o2_3) < 32 and 0 <= ((32 + o3_3)) < 64 else cutlass.BFloat16(0))
            load_3_3 = (tensor_bindings[2][o0_3, o1_3, o2_3, o3_3] if 0 <= (o0_3) < 1 and 0 <= (o1_3) < 1 and 0 <= (o2_3) < 32 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            load_4_3 = (tensor_bindings[2][o0_3, o1_3, o2_3, (-32 + o3_3)] if 0 <= (o0_3) < 1 and 0 <= (o1_3) < 1 and 0 <= (o2_3) < 32 and 0 <= ((-32 + o3_3)) < 64 else cutlass.BFloat16(0))
            load_5_3 = (tensor_bindings[0][o0_3, o1_3, o2_3, (-32 + o3_3)] if 0 <= (o0_3) < 1 and 0 <= (o1_3) < 1 and 0 <= (o2_3) < 32 and 0 <= ((-32 + o3_3)) < 64 else cutlass.BFloat16(0))
            load_6_3 = (tensor_bindings[1][o0_3, o1_3, o2_3, (-32 + o3_3)] if 0 <= (o0_3) < 1 and 0 <= (o1_3) < 1 and 0 <= (o2_3) < 32 and 0 <= ((-32 + o3_3)) < 64 else cutlass.BFloat16(0))
            value_0 = ((((((((load_0_0).to(cutlass.Float32) * (load_1_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) - ((((load_2_0).to(cutlass.Float32) * (load_3_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3_0 < cutlass.Int64(32))) else (((((((((load_0_0).to(cutlass.Float32) * (load_4_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + ((((load_5_0).to(cutlass.Float32) * (load_6_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3_0 < cutlass.Int64(64))) else (load_0_0))))
            value_1 = ((((((((load_0_1).to(cutlass.Float32) * (load_1_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) - ((((load_2_1).to(cutlass.Float32) * (load_3_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3_1 < cutlass.Int64(32))) else (((((((((load_0_1).to(cutlass.Float32) * (load_4_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + ((((load_5_1).to(cutlass.Float32) * (load_6_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3_1 < cutlass.Int64(64))) else (load_0_1))))
            value_2 = ((((((((load_0_2).to(cutlass.Float32) * (load_1_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) - ((((load_2_2).to(cutlass.Float32) * (load_3_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3_2 < cutlass.Int64(32))) else (((((((((load_0_2).to(cutlass.Float32) * (load_4_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + ((((load_5_2).to(cutlass.Float32) * (load_6_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3_2 < cutlass.Int64(64))) else (load_0_2))))
            value_3 = ((((((((load_0_3).to(cutlass.Float32) * (load_1_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) - ((((load_2_3).to(cutlass.Float32) * (load_3_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3_3 < cutlass.Int64(32))) else (((((((((load_0_3).to(cutlass.Float32) * (load_4_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + ((((load_5_3).to(cutlass.Float32) * (load_6_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3_3 < cutlass.Int64(64))) else (load_0_3))))
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_9(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        floats, counters, float_base, counter_base = epilogue
        current_length = tensor_bindings[5][0].to(cutlass.Int32)
        if current_length < 0:
            llvm.intr_trap()
        if current_length + cutlass.Int32(1) > cutlass.Int32(128):
            llvm.intr_trap()
        keys = current_length + cutlass.Int32(1)
        in_use = (keys + cutlass.Int32(63)) // cutlass.Int32(64)
        if in_use > cutlass.Int32(3):
            in_use = cutlass.Int32(3)
        chunk_keys = (keys + in_use - cutlass.Int32(1)) // in_use
        warp = thread // 32
        lane = thread % 32
        # Only the chunks in use are items, row-major so a row's items are
        # consecutive (see the deadlock note above).
        for item in cutlass.range(block, cutlass.Int32(32) * in_use, workers):
            bh = item // in_use
            chunk = item % in_use
            if True:
                b = bh // 32
                h = bh % 32
                kv_head = h // 1
                start = chunk * chunk_keys
                stop = start + chunk_keys
                if stop > keys:
                    stop = keys
                scores = float_base + bh * 128
                stats = float_base + 4096 + bh * 6
                # Cached keys come from the cache; the new key at current_length
                # (in the last chunk) from new_k and new_v.
                cached_stop = stop
                if cached_stop > current_length:
                    cached_stop = current_length
                # Every chunk in use runs the whole exchange, even with no keys.
                if True:
                    query = tensor_bindings[0][(b, 0, h, None)].load().to(cutlass.Float32)
                    # Scores: one key per thread, whole rows per load.
                    maximum = cutlass.Float32(-1.0e30)
                    for key in cutlass.range(start + thread, cached_stop, 256):
                        row = tensor_bindings[3][(b, key, kv_head, None)].load().to(cutlass.Float32)
                        score = cutlass.Float32((query * row).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0))
                        score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                        score = score * cutlass.Float32(0.125)
                        cute.arch.store(floats.iterator + scores + key, score, cop="cg")
                        maximum = cute.arch.fmax(maximum, score, nan=True)
                    if stop == keys:
                        if thread == 0:
                            row = tensor_bindings[1][(b, 0, kv_head, None)].load().to(cutlass.Float32)
                            score = cutlass.Float32((query * row).reduce(cute.ReductionOp.ADD, cutlass.Float32(0.0), 0))
                            score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                            score = score * cutlass.Float32(0.125)
                            cute.arch.store(floats.iterator + scores + current_length, score, cop="cg")
                            maximum = cute.arch.fmax(maximum, score, nan=True)
                    maximum = cute.arch.warp_reduction_max(maximum)
                    if lane == 0:
                        scratch[warp] = maximum
                    cute.arch.sync_threads()
                    if thread == 0:
                        chunk_maximum = scratch[0]
                        for other in cutlass.range_constexpr(1, 8):
                            chunk_maximum = cute.arch.fmax(chunk_maximum, scratch[other], nan=True)
                        cute.arch.store(floats.iterator + stats + chunk * 2, chunk_maximum, cop="cg")
                        scratch[0] = chunk_maximum
                    cute.arch.fence_acq_rel_gpu()
                    if in_use > cutlass.Int32(1):
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
                        probability = cute.exp(
                            cute.arch.load(floats.iterator + scores + key, cutlass.Float32, cop="cg") - row_maximum
                        )
                        cute.arch.store(floats.iterator + scores + key, probability, cop="cg")
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
                        cute.arch.store(floats.iterator + stats + chunk * 2 + 1, chunk_total, cop="cg")
                        scratch[0] = chunk_total
                    cute.arch.fence_acq_rel_gpu()
                    if in_use > cutlass.Int32(1):
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
                                cute.arch.load(floats.iterator + scores + key_0, cutlass.Float32, cop="cg") / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_0 = cached_stop - cutlass.Int32(1)
                        row_0_0 = tensor_bindings[4][b, key_0, kv_head, lane + 0].to(cutlass.Float32)
                        row_0_1 = tensor_bindings[4][b, key_0, kv_head, lane + 32].to(cutlass.Float32)
                        key_1 = base + 8
                        probability_1 = cutlass.Float32(0.0)
                        if key_1 < cached_stop:
                            probability_1 = (
                                cute.arch.load(floats.iterator + scores + key_1, cutlass.Float32, cop="cg") / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_1 = cached_stop - cutlass.Int32(1)
                        row_1_0 = tensor_bindings[4][b, key_1, kv_head, lane + 0].to(cutlass.Float32)
                        row_1_1 = tensor_bindings[4][b, key_1, kv_head, lane + 32].to(cutlass.Float32)
                        key_2 = base + 16
                        probability_2 = cutlass.Float32(0.0)
                        if key_2 < cached_stop:
                            probability_2 = (
                                cute.arch.load(floats.iterator + scores + key_2, cutlass.Float32, cop="cg") / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_2 = cached_stop - cutlass.Int32(1)
                        row_2_0 = tensor_bindings[4][b, key_2, kv_head, lane + 0].to(cutlass.Float32)
                        row_2_1 = tensor_bindings[4][b, key_2, kv_head, lane + 32].to(cutlass.Float32)
                        key_3 = base + 24
                        probability_3 = cutlass.Float32(0.0)
                        if key_3 < cached_stop:
                            probability_3 = (
                                cute.arch.load(floats.iterator + scores + key_3, cutlass.Float32, cop="cg") / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_3 = cached_stop - cutlass.Int32(1)
                        row_3_0 = tensor_bindings[4][b, key_3, kv_head, lane + 0].to(cutlass.Float32)
                        row_3_1 = tensor_bindings[4][b, key_3, kv_head, lane + 32].to(cutlass.Float32)
                        key_4 = base + 32
                        probability_4 = cutlass.Float32(0.0)
                        if key_4 < cached_stop:
                            probability_4 = (
                                cute.arch.load(floats.iterator + scores + key_4, cutlass.Float32, cop="cg") / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_4 = cached_stop - cutlass.Int32(1)
                        row_4_0 = tensor_bindings[4][b, key_4, kv_head, lane + 0].to(cutlass.Float32)
                        row_4_1 = tensor_bindings[4][b, key_4, kv_head, lane + 32].to(cutlass.Float32)
                        key_5 = base + 40
                        probability_5 = cutlass.Float32(0.0)
                        if key_5 < cached_stop:
                            probability_5 = (
                                cute.arch.load(floats.iterator + scores + key_5, cutlass.Float32, cop="cg") / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_5 = cached_stop - cutlass.Int32(1)
                        row_5_0 = tensor_bindings[4][b, key_5, kv_head, lane + 0].to(cutlass.Float32)
                        row_5_1 = tensor_bindings[4][b, key_5, kv_head, lane + 32].to(cutlass.Float32)
                        key_6 = base + 48
                        probability_6 = cutlass.Float32(0.0)
                        if key_6 < cached_stop:
                            probability_6 = (
                                cute.arch.load(floats.iterator + scores + key_6, cutlass.Float32, cop="cg") / denominator
                            ).to(cutlass.BFloat16).to(cutlass.Float32)
                        else:
                            key_6 = cached_stop - cutlass.Int32(1)
                        row_6_0 = tensor_bindings[4][b, key_6, kv_head, lane + 0].to(cutlass.Float32)
                        row_6_1 = tensor_bindings[4][b, key_6, kv_head, lane + 32].to(cutlass.Float32)
                        key_7 = base + 56
                        probability_7 = cutlass.Float32(0.0)
                        if key_7 < cached_stop:
                            probability_7 = (
                                cute.arch.load(floats.iterator + scores + key_7, cutlass.Float32, cop="cg") / denominator
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
                                cute.arch.load(floats.iterator + scores + current_length, cutlass.Float32, cop="cg") / denominator
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
                            tensor_bindings[6][bh, 0, d] = value.to(cutlass.BFloat16)
                    else:
                        partials = float_base + 4288 + bh * 192
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
                                tensor_bindings[6][bh, 0, d] = value.to(cutlass.BFloat16)
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
                    if h % 1 == 0:
                        if warp == 0:
                            for d in cutlass.range(lane, 64, 32):
                                tensor_bindings[3][b, current_length, kv_head, d] = tensor_bindings[1][b, 0, kv_head, d]
                                tensor_bindings[4][b, current_length, kv_head, d] = tensor_bindings[2][b, 0, kv_head, d]
                cute.arch.sync_threads()

    @cute.jit
    def _body_10(
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 2048
            remainder_0 = remainder_0 // 2048
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 2048 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[1][0, 0, o2_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 2048 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 2048
            remainder_1 = remainder_1 // 2048
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 2048 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[1][0, 0, o2_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 2048 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 2048
            remainder_2 = remainder_2 // 2048
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 2048 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[1][0, 0, o2_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 2048 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 2048
            remainder_3 = remainder_3 // 2048
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 2048 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[1][0, 0, o2_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 2048 else cutlass.BFloat16(0))
            value_0 = (((load_0_0).to(cutlass.Float32) + (load_1_0).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_1 = (((load_0_1).to(cutlass.Float32) + (load_1_1).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_2 = (((load_0_2).to(cutlass.Float32) + (load_1_2).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_3 = (((load_0_3).to(cutlass.Float32) + (load_1_3).to(cutlass.Float32))).to(cutlass.BFloat16)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3] = value_3

    @cute.jit
    def _body_12(
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 8192
            remainder_0 = remainder_0 // 8192
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 8192 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[1][0, 0, o2_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 8192 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 8192
            remainder_1 = remainder_1 // 8192
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 8192 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[1][0, 0, o2_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 8192 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 8192
            remainder_2 = remainder_2 // 8192
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 8192 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[1][0, 0, o2_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 8192 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 8192
            remainder_3 = remainder_3 // 8192
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 8192 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[1][0, 0, o2_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 8192 else cutlass.BFloat16(0))
            value_0 = ((((((load_0_0).to(cutlass.Float32) / (cute.exp((-((load_0_0).to(cutlass.Float32)))) + (cutlass.Int64(1)).to(cutlass.Float32)))).to(cutlass.BFloat16)).to(cutlass.Float32) * (load_1_0).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_1 = ((((((load_0_1).to(cutlass.Float32) / (cute.exp((-((load_0_1).to(cutlass.Float32)))) + (cutlass.Int64(1)).to(cutlass.Float32)))).to(cutlass.BFloat16)).to(cutlass.Float32) * (load_1_1).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_2 = ((((((load_0_2).to(cutlass.Float32) / (cute.exp((-((load_0_2).to(cutlass.Float32)))) + (cutlass.Int64(1)).to(cutlass.Float32)))).to(cutlass.BFloat16)).to(cutlass.Float32) * (load_1_2).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_3 = ((((((load_0_3).to(cutlass.Float32) / (cute.exp((-((load_0_3).to(cutlass.Float32)))) + (cutlass.Int64(1)).to(cutlass.Float32)))).to(cutlass.BFloat16)).to(cutlass.Float32) * (load_1_3).to(cutlass.Float32))).to(cutlass.BFloat16)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3] = value_3


    # Publish this call's boundary addresses before any phase reads them.
    if _block == 0:
        _dynamic = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int64,
                dynamic_addresses + cutlass.Int64(_call % 64) * 7760,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            cute.make_layout(970),
        )
        if _dynamic[0] != cutlass.Int64(_call):
            llvm.intr_trap()
        for _position in cutlass.range(_thread, 969, 256, unroll=1):
            addresses[dynamic_slot_table[_position]] = _dynamic[_position + 1]

    counting_grid_barrier(ready, workers, _ordinal)
    if _block == 0:
        if _thread == 0:
            ready[workers + 2] = _call + cutlass.Int32(1)
    _warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if _warp == 0:
        for gemm in cutlass.range(_block, 169, workers, unroll=1):
            gemm_variant = gemm_table[gemm, 0]
            if gemm_variant == 0:
                build_table_descriptors(
                    gemm_values[0],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (2048, 1, 2048),
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
                    (8192, 1, 2048),
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
                    (2048, 1, 8192),
                    True,
                    1,
                )
            if gemm_variant == 3:
                build_table_descriptors(
                    gemm_values[3],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (49152, 1, 2048),
                    True,
                    1,
                )
        cute.nvgpu.cpasync.fence_tma_desc_release()
    counting_grid_barrier(ready, workers, _ordinal + cutlass.Int32(1))

    for phase in cutlass.range(483, unroll=1):
        variant = phase_table[phase, 0]
        _rotated = (_block + workers - phase_table[phase, 2]) % workers
        _workers = workers
        if phase_table[phase, 2] < 0:
            # A recompute phase: this CTA runs every tile itself.
            _rotated = cutlass.Int32(0)
            _workers = cutlass.Int32(1)
        if variant == 0:
            _body_0(
                work_iterators[0],
                _rotated,
                _thread,
                _workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1), stride=(1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((49152, 2048), stride=(2048, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 2048), stride=(2048, 2048, 1)), 2),),
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
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 2048), stride=(2048, 2048, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((2048,), stride=(1,)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 2048), stride=(2048, 2048, 1)), 2),),
                _scratch_25,
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
            _body_6(
                work_iterators[5],
                _rotated,
                _thread,
                _workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 7:
            _body_7(
                work_iterators[6],
                _rotated,
                _thread,
                _workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 8:
            _body_8(
                work_iterators[7],
                _rotated,
                _thread,
                _workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 32, 64), stride=(2048, 2048, 64, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 32, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 32, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 1, 32, 64), stride=(2048, 2048, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 9:
            _body_9(
                work_iterators[8],
                _rotated,
                _thread,
                _workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 32, 64), stride=(2048, 2048, 64, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 32, 64), stride=(2048, 2048, 64, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 32, 64), stride=(2048, 2048, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 128, 32, 64), stride=(262144, 2048, 64, 1)), 2), _slot(addresses, phase_table, phase, 4, cutlass.BFloat16, cute.make_layout((1, 128, 32, 64), stride=(262144, 2048, 64, 1)), 2), _slot(addresses, phase_table, phase, 5, cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4), _slot(addresses, phase_table, phase, 6, cutlass.BFloat16, cute.make_layout((32, 1, 64), stride=(64, 64, 1)), 2),),
                _scratch_127,
                (phase_floats, phase_counters, phase_table[phase, 4], phase_table[phase, 5]),
            )
        if variant == 10:
            _body_10(
                work_iterators[9],
                _rotated,
                _thread,
                _workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 2048), stride=(2048, 2048, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 2048), stride=(2048, 2048, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 2048), stride=(2048, 2048, 1)), 2),),
                None,
                None,
            )
        if variant == 11:
            _gemm_1.run_table(
                gemm_values[1],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 6]),
                _rotated,
            )
        if variant == 12:
            _body_12(
                work_iterators[10],
                _rotated,
                _thread,
                _workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 8192), stride=(8192, 8192, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 8192), stride=(8192, 8192, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 8192), stride=(8192, 8192, 1)), 2),),
                None,
                None,
            )
        if variant == 13:
            _gemm_2.run_table(
                gemm_values[2],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 6]),
                _rotated,
            )
        if variant == 14:
            _gemm_3.run_table(
                gemm_values[3],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 6]),
                _rotated,
            )
        if phase_table[phase, 2] < 0:
            # Later TMA reads in this interval see the generic-proxy writes.
            cute.arch.fence_proxy("async.global")
        if phase_table[phase, 1] != 0:
            counting_grid_barrier(ready, workers, _ordinal + phase_table[phase, 1])
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
        tma_descriptors,
        gemm_values,
    ).launch(
        grid=[1, 1, workers],
        block=[256, 1, 1],
        cooperative=True,
        stream=stream,
    )

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cute.tensor import _Tensor

from megabake.runtime import counting_grid_barrier, prefetch_gemm_weights, prefetch_l1
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
    bias_features=896,
    bias_along_m=True,
    direct_store=True,
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
    bias_features=128,
    bias_along_m=True,
    direct_store=True,
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
    direct_store=True,
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
)
_gemm_5 = HopperWgmmaGemmPersistentKernel(
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
    address = addresses[1577 + phase * 5 + column]
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
class _Phase9Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase26Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 64],
        16
    ]


@cute.union
class _WorkerScratch:
    phase_9: _Phase9Scratch
    phase_26: _Phase26Scratch

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
    _ordinal = _call * cutlass.Int32(292)
    _worker_scratch = cutlass.utils.SmemAllocator().allocate(_WorkerScratch)
    _scratch_9 = _worker_scratch.phase_9.body.get_tensor(cute.make_layout(8))
    _scratch_26 = _worker_scratch.phase_26.body.get_tensor(cute.make_layout(64))
    if _thread == 0:
        _worker_scratch.phase_9.barrier_0[0] = 0
    cute.arch.sync_threads()

    _gemm_scratch = cutlass.utils.SmemAllocator().allocate(_gemm_0.shared_storage)

    @cute.jit
    def _body_0(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[0]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o0_0 = remainder_0 % 4
            linear_1 = base + 256
            remainder_1 = linear_1
            o0_1 = remainder_1 % 4
            linear_2 = base + 512
            remainder_2 = linear_2
            o0_2 = remainder_2 % 4
            linear_3 = base + 768
            remainder_3 = linear_3
            o0_3 = remainder_3 % 4
            value_0 = (o0_0 + cutlass.Int64(0))
            value_1 = (o0_1 + cutlass.Int64(0))
            value_2 = (o0_2 + cutlass.Int64(0))
            value_3 = (o0_3 + cutlass.Int64(0))
            if linear_0 < output_count:
                output[o0_0] = value_0
            if linear_1 < output_count:
                output[o0_1] = value_1
            if linear_2 < output_count:
                output[o0_2] = value_2
            if linear_3 < output_count:
                output[o0_3] = value_3

    @cute.jit
    def _body_1(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[0]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            linear_1 = base + 256
            remainder_1 = linear_1
            linear_2 = base + 512
            remainder_2 = linear_2
            linear_3 = base + 768
            remainder_3 = linear_3
            value_0 = cutlass.BFloat16(0.0)
            value_1 = cutlass.BFloat16(0.0)
            value_2 = cutlass.BFloat16(0.0)
            value_3 = cutlass.BFloat16(0.0)
            if linear_0 < output_count:
                output[0] = value_0
            if linear_1 < output_count:
                output[0] = value_1
            if linear_2 < output_count:
                output[0] = value_2
            if linear_3 < output_count:
                output[0] = value_3

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
        output = tensor_bindings[2]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 4, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 896
            remainder_0 = remainder_0 // 896
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 else cutlass.Int64(0))
            if load_0_0 < 0:
                llvm.intr_trap()
            if load_0_0 >= 129280:
                llvm.intr_trap()
            load_1_0 = (tensor_bindings[1][(tensor_bindings[0][0, o1_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 else cutlass.Int64(0)), o2_0] if 0 <= (o2_0) < 896 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 896
            remainder_1 = remainder_1 // 896
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 else cutlass.Int64(0))
            if load_0_1 < 0:
                llvm.intr_trap()
            if load_0_1 >= 129280:
                llvm.intr_trap()
            load_1_1 = (tensor_bindings[1][(tensor_bindings[0][0, o1_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 else cutlass.Int64(0)), o2_1] if 0 <= (o2_1) < 896 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 896
            remainder_2 = remainder_2 // 896
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 else cutlass.Int64(0))
            if load_0_2 < 0:
                llvm.intr_trap()
            if load_0_2 >= 129280:
                llvm.intr_trap()
            load_1_2 = (tensor_bindings[1][(tensor_bindings[0][0, o1_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 else cutlass.Int64(0)), o2_2] if 0 <= (o2_2) < 896 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 896
            remainder_3 = remainder_3 // 896
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 else cutlass.Int64(0))
            if load_0_3 < 0:
                llvm.intr_trap()
            if load_0_3 >= 129280:
                llvm.intr_trap()
            load_1_3 = (tensor_bindings[1][(tensor_bindings[0][0, o1_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 else cutlass.Int64(0)), o2_3] if 0 <= (o2_3) < 896 else cutlass.BFloat16(0))
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
    def _body_3(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[0]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            linear_1 = base + 256
            remainder_1 = linear_1
            linear_2 = base + 512
            remainder_2 = linear_2
            linear_3 = base + 768
            remainder_3 = linear_3
            value_0 = cutlass.Boolean(True)
            value_1 = cutlass.Boolean(True)
            value_2 = cutlass.Boolean(True)
            value_3 = cutlass.Boolean(True)
            if linear_0 < output_count:
                output[0] = value_0
            if linear_1 < output_count:
                output[0] = value_1
            if linear_2 < output_count:
                output[0] = value_2
            if linear_3 < output_count:
                output[0] = value_3

    @cute.jit
    def _body_4(
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
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 else cutlass.Int64(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 else cutlass.Int64(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 else cutlass.Int64(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 else cutlass.Int64(0))
            value_0 = (load_0_0).to(cutlass.Boolean)
            value_1 = (load_0_1).to(cutlass.Boolean)
            value_2 = (load_0_2).to(cutlass.Boolean)
            value_3 = (load_0_3).to(cutlass.Boolean)
            if linear_0 < output_count:
                output[o0_0, o1_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3] = value_3

    @cute.jit
    def _body_5(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[0]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o0_0 = remainder_0 % 1
            linear_1 = base + 256
            remainder_1 = linear_1
            o0_1 = remainder_1 % 1
            linear_2 = base + 512
            remainder_2 = linear_2
            o0_2 = remainder_2 % 1
            linear_3 = base + 768
            remainder_3 = linear_3
            o0_3 = remainder_3 % 1
            value_0 = o0_0
            value_1 = o0_1
            value_2 = o0_2
            value_3 = o0_3
            if linear_0 < output_count:
                output[o0_0] = value_0
            if linear_1 < output_count:
                output[o0_1] = value_1
            if linear_2 < output_count:
                output[o0_2] = value_2
            if linear_3 < output_count:
                output[o0_3] = value_3

    @cute.jit
    def _body_6(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor],
        scratch,
        epilogue,
    ):
        output = tensor_bindings[0]
        output_count = cute.size(output)
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            linear_1 = base + 256
            remainder_1 = linear_1
            linear_2 = base + 512
            remainder_2 = linear_2
            linear_3 = base + 768
            remainder_3 = linear_3
            value_0 = cutlass.BFloat16(-3.3895313892515355e+38)
            value_1 = cutlass.BFloat16(-3.3895313892515355e+38)
            value_2 = cutlass.BFloat16(-3.3895313892515355e+38)
            value_3 = cutlass.BFloat16(-3.3895313892515355e+38)
            if linear_0 < output_count:
                output[0] = value_0
            if linear_1 < output_count:
                output[0] = value_1
            if linear_2 < output_count:
                output[0] = value_2
            if linear_3 < output_count:
                output[0] = value_3

    @cute.jit
    def _body_7(
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
        for tile in cutlass.range(block, 4, workers):
            row_index = (cutlass.Int64(tile) * 1)
            if row_index < 4:
                o1 = (row_index // 1) % 4
                o0 = (row_index // 4) % 1
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for reduction_index in cutlass.range(thread, 896, 256, unroll_full=True):
                    r0 = reduction_index
                    accumulator = (accumulator + cute.pow(((tensor_bindings[0][o0, o1, r0] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (r0) < 896 else cutlass.BFloat16(0))).to(cutlass.Float32), (cutlass.Int64(2)).to(cutlass.Float32)))
                warp_partial = cute.arch.warp_reduction_sum(accumulator)
                if lane == 0:
                    scratch[ warp] = warp_partial
                cute.arch.sync_threads()
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for warp_index in cutlass.range(warps):
                    accumulator = accumulator + scratch[ warp_index]
                post_index_0 = thread + 0
                o2_0 = (post_index_0 // 1) % 896
                post_load_0_0 = (tensor_bindings[0][o0, o1, o2_0] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2_0) < 896 else cutlass.BFloat16(0))
                post_load_1_0 = (tensor_bindings[1][o2_0] if 0 <= (o2_0) < 896 else cutlass.BFloat16(0))
                post_index_1 = thread + 256
                o2_1 = (post_index_1 // 1) % 896
                post_load_0_1 = (tensor_bindings[0][o0, o1, o2_1] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2_1) < 896 else cutlass.BFloat16(0))
                post_load_1_1 = (tensor_bindings[1][o2_1] if 0 <= (o2_1) < 896 else cutlass.BFloat16(0))
                post_index_2 = thread + 512
                o2_2 = (post_index_2 // 1) % 896
                post_load_0_2 = (tensor_bindings[0][o0, o1, o2_2] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2_2) < 896 else cutlass.BFloat16(0))
                post_load_1_2 = (tensor_bindings[1][o2_2] if 0 <= (o2_2) < 896 else cutlass.BFloat16(0))
                post_index_3 = thread + 768
                o2_3 = (post_index_3 // 1) % 896
                post_load_0_3 = (tensor_bindings[0][o0, o1, o2_3] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2_3) < 896 else cutlass.BFloat16(0))
                post_load_1_3 = (tensor_bindings[1][o2_3] if 0 <= (o2_3) < 896 else cutlass.BFloat16(0))
                output[o0, o1, o2_0] = (((post_load_1_0).to(cutlass.Float32) * ((((post_load_0_0).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(896)).to(cutlass.Float32)) + cutlass.Float32(1e-06))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_1] = (((post_load_1_1).to(cutlass.Float32) * ((((post_load_0_1).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(896)).to(cutlass.Float32)) + cutlass.Float32(1e-06))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2_2] = (((post_load_1_2).to(cutlass.Float32) * ((((post_load_0_2).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(896)).to(cutlass.Float32)) + cutlass.Float32(1e-06))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                if post_index_3 < 896:
                    output[o0, o1, o2_3] = (((post_load_1_3).to(cutlass.Float32) * ((((post_load_0_3).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(896)).to(cutlass.Float32)) + cutlass.Float32(1e-06))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                cute.arch.sync_threads()

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
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o2_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, 0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o2_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, 0] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o2_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, 0] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o2_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, 0] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
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
    def _body_9(
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
            o3_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, 0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_0) < 4 else cutlass.Int64(0))
            load_1_0 = (tensor_bindings[1][0, 0, o2_0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, 0, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_1) < 4 else cutlass.Int64(0))
            load_1_1 = (tensor_bindings[1][0, 0, o2_1, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, 0, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_2) < 4 else cutlass.Int64(0))
            load_1_2 = (tensor_bindings[1][0, 0, o2_2, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, 0, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_3) < 4 else cutlass.Int64(0))
            load_1_3 = (tensor_bindings[1][0, 0, o2_3, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            value_0 = (load_0_0 <= load_1_0)
            value_1 = (load_0_1 <= load_1_1)
            value_2 = (load_0_2 <= load_1_2)
            value_3 = (load_0_3 <= load_1_3)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_10(
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
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o2_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            load_1_0 = (tensor_bindings[1][0, 0, 0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_0) < 4 else cutlass.Int64(0))
            if load_0_0 < 0:
                llvm.intr_trap()
            if load_0_0 >= 1:
                llvm.intr_trap()
            if load_1_0 < 0:
                llvm.intr_trap()
            if load_1_0 >= 4:
                llvm.intr_trap()
            load_2_0 = tensor_bindings[2][(tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), (tensor_bindings[1][0, 0, 0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_0) < 4 else cutlass.Int64(0))]
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o2_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            load_1_1 = (tensor_bindings[1][0, 0, 0, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_1) < 4 else cutlass.Int64(0))
            if load_0_1 < 0:
                llvm.intr_trap()
            if load_0_1 >= 1:
                llvm.intr_trap()
            if load_1_1 < 0:
                llvm.intr_trap()
            if load_1_1 >= 4:
                llvm.intr_trap()
            load_2_1 = tensor_bindings[2][(tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), (tensor_bindings[1][0, 0, 0, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_1) < 4 else cutlass.Int64(0))]
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o2_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            load_1_2 = (tensor_bindings[1][0, 0, 0, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_2) < 4 else cutlass.Int64(0))
            if load_0_2 < 0:
                llvm.intr_trap()
            if load_0_2 >= 1:
                llvm.intr_trap()
            if load_1_2 < 0:
                llvm.intr_trap()
            if load_1_2 >= 4:
                llvm.intr_trap()
            load_2_2 = tensor_bindings[2][(tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), (tensor_bindings[1][0, 0, 0, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_2) < 4 else cutlass.Int64(0))]
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o2_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
            load_1_3 = (tensor_bindings[1][0, 0, 0, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_3) < 4 else cutlass.Int64(0))
            if load_0_3 < 0:
                llvm.intr_trap()
            if load_0_3 >= 1:
                llvm.intr_trap()
            if load_1_3 < 0:
                llvm.intr_trap()
            if load_1_3 >= 4:
                llvm.intr_trap()
            load_2_3 = tensor_bindings[2][(tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), (tensor_bindings[1][0, 0, 0, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_3) < 4 else cutlass.Int64(0))]
            value_0 = load_2_0
            value_1 = load_2_1
            value_2 = load_2_2
            value_3 = load_2_3
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_11(
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
            load_0_0 = tensor_bindings[0][0]
            linear_1 = base + 256
            remainder_1 = linear_1
            load_0_1 = tensor_bindings[0][0]
            linear_2 = base + 512
            remainder_2 = linear_2
            load_0_2 = tensor_bindings[0][0]
            linear_3 = base + 768
            remainder_3 = linear_3
            load_0_3 = tensor_bindings[0][0]
            value_0 = load_0_0
            value_1 = load_0_1
            value_2 = load_0_2
            value_3 = load_0_3
            if linear_0 < output_count:
                output[0] = value_0
            if linear_1 < output_count:
                output[0] = value_1
            if linear_2 < output_count:
                output[0] = value_2
            if linear_3 < output_count:
                output[0] = value_3

    @cute.jit
    def _body_13(
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
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, 0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_0 = (tensor_bindings[1][o2_0] if 0 <= (o2_0) < 32 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 32
            remainder_1 = remainder_1 // 32
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, 0] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_1 = (tensor_bindings[1][o2_1] if 0 <= (o2_1) < 32 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 32
            remainder_2 = remainder_2 // 32
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, 0] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_2 = (tensor_bindings[1][o2_2] if 0 <= (o2_2) < 32 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 32
            remainder_3 = remainder_3 // 32
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, 0] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
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
    def _body_15(
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
            o3_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = tensor_bindings[0][0]
            load_1_0 = (tensor_bindings[1][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 4 else cutlass.Boolean(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = tensor_bindings[0][0]
            load_1_1 = (tensor_bindings[1][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 4 else cutlass.Boolean(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = tensor_bindings[0][0]
            load_1_2 = (tensor_bindings[1][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 4 else cutlass.Boolean(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = tensor_bindings[0][0]
            load_1_3 = (tensor_bindings[1][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 4 else cutlass.Boolean(0))
            value_0 = (load_0_0 & load_1_0)
            value_1 = (load_0_1 & load_1_1)
            value_2 = (load_0_2 & load_1_2)
            value_3 = (load_0_3 & load_1_3)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_16(
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
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 2 and 0 <= (o3_0) < 32 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 32
            remainder_1 = remainder_1 // 32
            o2_1 = remainder_1 % 2
            remainder_1 = remainder_1 // 2
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 2 and 0 <= (o3_1) < 32 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 32
            remainder_2 = remainder_2 // 32
            o2_2 = remainder_2 % 2
            remainder_2 = remainder_2 // 2
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 2 and 0 <= (o3_2) < 32 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 32
            remainder_3 = remainder_3 // 32
            o2_3 = remainder_3 % 2
            remainder_3 = remainder_3 // 2
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 2 and 0 <= (o3_3) < 32 else cutlass.Float32(0))
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
    def _body_17(
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
            o3_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 4 else cutlass.Boolean(0))
            load_1_0 = (tensor_bindings[1][0, 0, 0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_0) < 4 else cutlass.Boolean(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 4 else cutlass.Boolean(0))
            load_1_1 = (tensor_bindings[1][0, 0, 0, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_1) < 4 else cutlass.Boolean(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 4 else cutlass.Boolean(0))
            load_1_2 = (tensor_bindings[1][0, 0, 0, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_2) < 4 else cutlass.Boolean(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 4 else cutlass.Boolean(0))
            load_1_3 = (tensor_bindings[1][0, 0, 0, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_3) < 4 else cutlass.Boolean(0))
            value_0 = (load_0_0 & load_1_0)
            value_1 = (load_0_1 & load_1_1)
            value_2 = (load_0_2 & load_1_2)
            value_3 = (load_0_3 & load_1_3)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_18(
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
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 64 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 64 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 64 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 64 else cutlass.Float32(0))
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
    def _body_19(
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
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 64 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 64 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 64 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 64 else cutlass.Float32(0))
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
    def _body_20(
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
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 4 else cutlass.Boolean(0))
            load_1_0 = tensor_bindings[1][0]
            load_2_0 = tensor_bindings[2][0]
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 4 else cutlass.Boolean(0))
            load_1_1 = tensor_bindings[1][0]
            load_2_1 = tensor_bindings[2][0]
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 4 else cutlass.Boolean(0))
            load_1_2 = tensor_bindings[1][0]
            load_2_2 = tensor_bindings[2][0]
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 4 else cutlass.Boolean(0))
            load_1_3 = tensor_bindings[1][0]
            load_2_3 = tensor_bindings[2][0]
            value_0 = ((load_1_0) if (load_0_0) else (load_2_0))
            value_1 = ((load_1_1) if (load_0_1) else (load_2_1))
            value_2 = ((load_1_2) if (load_0_2) else (load_2_2))
            value_3 = ((load_1_3) if (load_0_3) else (load_2_3))
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_21(
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
        for tile in cutlass.range(block, 4, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 64
            remainder_0 = remainder_0 // 64
            o1_0 = remainder_0 % 14
            remainder_0 = remainder_0 // 14
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0, (32 + o3_0)] if 0 <= (0) < 1 and 0 <= (o1_0) < 14 and 0 <= (o2_0) < 4 and 0 <= ((32 + o3_0)) < 64 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[0][0, o1_0, o2_0, (-32 + o3_0)] if 0 <= (0) < 1 and 0 <= (o1_0) < 14 and 0 <= (o2_0) < 4 and 0 <= ((-32 + o3_0)) < 64 else cutlass.BFloat16(0))
            load_2_0 = (tensor_bindings[1][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            load_3_0 = (tensor_bindings[0][0, o1_0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 14 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            load_4_0 = (tensor_bindings[2][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o1_1 = remainder_1 % 14
            remainder_1 = remainder_1 // 14
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1, (32 + o3_1)] if 0 <= (0) < 1 and 0 <= (o1_1) < 14 and 0 <= (o2_1) < 4 and 0 <= ((32 + o3_1)) < 64 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[0][0, o1_1, o2_1, (-32 + o3_1)] if 0 <= (0) < 1 and 0 <= (o1_1) < 14 and 0 <= (o2_1) < 4 and 0 <= ((-32 + o3_1)) < 64 else cutlass.BFloat16(0))
            load_2_1 = (tensor_bindings[1][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            load_3_1 = (tensor_bindings[0][0, o1_1, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 14 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            load_4_1 = (tensor_bindings[2][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o1_2 = remainder_2 % 14
            remainder_2 = remainder_2 // 14
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2, (32 + o3_2)] if 0 <= (0) < 1 and 0 <= (o1_2) < 14 and 0 <= (o2_2) < 4 and 0 <= ((32 + o3_2)) < 64 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[0][0, o1_2, o2_2, (-32 + o3_2)] if 0 <= (0) < 1 and 0 <= (o1_2) < 14 and 0 <= (o2_2) < 4 and 0 <= ((-32 + o3_2)) < 64 else cutlass.BFloat16(0))
            load_2_2 = (tensor_bindings[1][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            load_3_2 = (tensor_bindings[0][0, o1_2, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 14 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            load_4_2 = (tensor_bindings[2][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o1_3 = remainder_3 % 14
            remainder_3 = remainder_3 // 14
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3, (32 + o3_3)] if 0 <= (0) < 1 and 0 <= (o1_3) < 14 and 0 <= (o2_3) < 4 and 0 <= ((32 + o3_3)) < 64 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[0][0, o1_3, o2_3, (-32 + o3_3)] if 0 <= (0) < 1 and 0 <= (o1_3) < 14 and 0 <= (o2_3) < 4 and 0 <= ((-32 + o3_3)) < 64 else cutlass.BFloat16(0))
            load_2_3 = (tensor_bindings[1][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            load_3_3 = (tensor_bindings[0][0, o1_3, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 14 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            load_4_3 = (tensor_bindings[2][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            value_0 = ((((((load_3_0).to(cutlass.Float32) * (load_4_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_0))) if ((o3_0 < cutlass.Int64(32))) else (load_1_0))).to(cutlass.Float32) * (load_2_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_1 = ((((((load_3_1).to(cutlass.Float32) * (load_4_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_1))) if ((o3_1 < cutlass.Int64(32))) else (load_1_1))).to(cutlass.Float32) * (load_2_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_2 = ((((((load_3_2).to(cutlass.Float32) * (load_4_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_2))) if ((o3_2 < cutlass.Int64(32))) else (load_1_2))).to(cutlass.Float32) * (load_2_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_3 = ((((((load_3_3).to(cutlass.Float32) * (load_4_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_3))) if ((o3_3 < cutlass.Int64(32))) else (load_1_3))).to(cutlass.Float32) * (load_2_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_22(
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
        for tile in cutlass.range(block, 1, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 64
            remainder_0 = remainder_0 // 64
            o1_0 = remainder_0 % 2
            remainder_0 = remainder_0 // 2
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0, (32 + o3_0)] if 0 <= (0) < 1 and 0 <= (o1_0) < 2 and 0 <= (o2_0) < 4 and 0 <= ((32 + o3_0)) < 64 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[0][0, o1_0, o2_0, (-32 + o3_0)] if 0 <= (0) < 1 and 0 <= (o1_0) < 2 and 0 <= (o2_0) < 4 and 0 <= ((-32 + o3_0)) < 64 else cutlass.BFloat16(0))
            load_2_0 = (tensor_bindings[1][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            load_3_0 = (tensor_bindings[0][0, o1_0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 2 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            load_4_0 = (tensor_bindings[2][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o1_1 = remainder_1 % 2
            remainder_1 = remainder_1 // 2
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1, (32 + o3_1)] if 0 <= (0) < 1 and 0 <= (o1_1) < 2 and 0 <= (o2_1) < 4 and 0 <= ((32 + o3_1)) < 64 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[0][0, o1_1, o2_1, (-32 + o3_1)] if 0 <= (0) < 1 and 0 <= (o1_1) < 2 and 0 <= (o2_1) < 4 and 0 <= ((-32 + o3_1)) < 64 else cutlass.BFloat16(0))
            load_2_1 = (tensor_bindings[1][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            load_3_1 = (tensor_bindings[0][0, o1_1, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 2 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            load_4_1 = (tensor_bindings[2][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o1_2 = remainder_2 % 2
            remainder_2 = remainder_2 // 2
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2, (32 + o3_2)] if 0 <= (0) < 1 and 0 <= (o1_2) < 2 and 0 <= (o2_2) < 4 and 0 <= ((32 + o3_2)) < 64 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[0][0, o1_2, o2_2, (-32 + o3_2)] if 0 <= (0) < 1 and 0 <= (o1_2) < 2 and 0 <= (o2_2) < 4 and 0 <= ((-32 + o3_2)) < 64 else cutlass.BFloat16(0))
            load_2_2 = (tensor_bindings[1][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            load_3_2 = (tensor_bindings[0][0, o1_2, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 2 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            load_4_2 = (tensor_bindings[2][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o1_3 = remainder_3 % 2
            remainder_3 = remainder_3 // 2
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3, (32 + o3_3)] if 0 <= (0) < 1 and 0 <= (o1_3) < 2 and 0 <= (o2_3) < 4 and 0 <= ((32 + o3_3)) < 64 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[0][0, o1_3, o2_3, (-32 + o3_3)] if 0 <= (0) < 1 and 0 <= (o1_3) < 2 and 0 <= (o2_3) < 4 and 0 <= ((-32 + o3_3)) < 64 else cutlass.BFloat16(0))
            load_2_3 = (tensor_bindings[1][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            load_3_3 = (tensor_bindings[0][0, o1_3, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 2 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            load_4_3 = (tensor_bindings[2][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
            value_0 = ((((((load_3_0).to(cutlass.Float32) * (load_4_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_0))) if ((o3_0 < cutlass.Int64(32))) else (load_1_0))).to(cutlass.Float32) * (load_2_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_1 = ((((((load_3_1).to(cutlass.Float32) * (load_4_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_1))) if ((o3_1 < cutlass.Int64(32))) else (load_1_1))).to(cutlass.Float32) * (load_2_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_2 = ((((((load_3_2).to(cutlass.Float32) * (load_4_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_2))) if ((o3_2 < cutlass.Int64(32))) else (load_1_2))).to(cutlass.Float32) * (load_2_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_3 = ((((((load_3_3).to(cutlass.Float32) * (load_4_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_3))) if ((o3_3 < cutlass.Int64(32))) else (load_1_3))).to(cutlass.Float32) * (load_2_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_23(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        # Each CTA tile covers 8 query rows, one per warp: lanes split the keys
        # for scores and the head dimensions for the value sum, with the row's
        # scores and probabilities in that warp's scratch slice. Rounding points
        # match the captured program; the value sum runs in key order.
        warp = thread // 32
        lane = thread % 32
        row_count = 56
        for tile in cutlass.range(block, 7, workers):
            row = (cutlass.Int64(tile) * 1) * 8 + warp
            if warp < 8:
                if row < row_count:
                    bh = row // 4
                    s = row % 4
                    b = bh // 14
                    h = bh % 14
                    kv_head = h // 7
                    base = warp * 4
                    maximum = cutlass.Float32(-1.0e30)
                    for key in cutlass.range(lane, 4, 32):
                        score = cutlass.Float32(0.0)
                        for d in cutlass.range_constexpr(64):
                            score = score + tensor_bindings[0][b, h, s, d].to(cutlass.Float32) * tensor_bindings[1][b, kv_head, key, d].to(cutlass.Float32)
                        score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                        score = (score * cutlass.Float32(0.125)).to(cutlass.BFloat16)
                        score = (score.to(cutlass.Float32) + cutlass.Float32(-3.3895313892515355e+38)).to(cutlass.BFloat16) if not tensor_bindings[3][b, 0, s, key] else score
                        score = score.to(cutlass.Float32)
                        scratch[base + key] = score
                        maximum = cute.arch.fmax(maximum, score, nan=True)
                    maximum = cute.arch.warp_reduction_max(maximum)
                    total = cutlass.Float32(0.0)
                    for key in cutlass.range(lane, 4, 32):
                        probability = cute.exp(scratch[base + key] - maximum)
                        scratch[base + key] = probability
                        total = total + probability
                    total = cute.arch.warp_reduction_sum(total)
                    denominator = total
                    if total <= cutlass.Float32(0.0):
                        denominator = cutlass.Float32(1.0)
                    for key in cutlass.range(lane, 4, 32):
                        scratch[base + key] = (scratch[base + key] / denominator).to(cutlass.BFloat16).to(cutlass.Float32)
                    cute.arch.sync_warp()
                    for d in cutlass.range(lane, 64, 32):
                        value = cutlass.Float32(0.0)
                        for key in cutlass.range(4, unroll=8):
                            value = value + scratch[base + key] * tensor_bindings[2][b, key, kv_head, d].to(cutlass.Float32)
                        tensor_bindings[4][bh, s, d] = value.to(cutlass.BFloat16)
                    cute.arch.sync_warp()

    @cute.jit
    def _body_24(
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
        for tile in cutlass.range(block, 4, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 64
            remainder_0 = remainder_0 // 64
            o2_0 = remainder_0 % 14
            remainder_0 = remainder_0 // 14
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 14 and 0 <= (o3_0) < 64 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o2_1 = remainder_1 % 14
            remainder_1 = remainder_1 // 14
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 14 and 0 <= (o3_1) < 64 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o2_2 = remainder_2 % 14
            remainder_2 = remainder_2 // 14
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 14 and 0 <= (o3_2) < 64 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o2_3 = remainder_3 % 14
            remainder_3 = remainder_3 // 14
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 14 and 0 <= (o3_3) < 64 else cutlass.BFloat16(0))
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
    def _body_26(
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
        for tile in cutlass.range(block, 4, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 896
            remainder_0 = remainder_0 // 896
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 896 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[1][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 896 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 896
            remainder_1 = remainder_1 // 896
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 896 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[1][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 896 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 896
            remainder_2 = remainder_2 // 896
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 896 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[1][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 896 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 896
            remainder_3 = remainder_3 // 896
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 896 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[1][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 896 else cutlass.BFloat16(0))
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
    def _body_28(
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
        for tile in cutlass.range(block, 19, workers):
            base = (cutlass.Int64(tile) * 1024) + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 4864
            remainder_0 = remainder_0 // 4864
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 4864 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[1][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 4864 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 4864
            remainder_1 = remainder_1 // 4864
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 4864 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[1][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 4864 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 4864
            remainder_2 = remainder_2 // 4864
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 4864 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[1][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 4864 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 4864
            remainder_3 = remainder_3 // 4864
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 4864 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[1][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 4864 else cutlass.BFloat16(0))
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
                dynamic_addresses + cutlass.Int64(_call % 64) * 9856,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            cute.make_layout(1232),
        )
        if _dynamic[0] != cutlass.Int64(_call):
            llvm.intr_trap()
        for _position in cutlass.range(_thread, 1231, 256, unroll=1):
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
                    (896, 4, 896),
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
                    (128, 4, 896),
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
                    (896, 4, 896),
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
                    (4864, 4, 896),
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
                    (896, 4, 4864),
                    True,
                    1,
                )
            if gemm_variant == 5:
                build_table_descriptors(
                    gemm_values[5],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (129280, 4, 896),
                    True,
                    1,
                )
        cute.nvgpu.cpasync.fence_tma_desc_release()
    counting_grid_barrier(ready, workers, _ordinal + cutlass.Int32(1))

    for phase in cutlass.range(406, unroll=1):
        if _thread == 0:
            if phase + 1 < 406:
                prefetch_l1(phase_table.iterator + (phase + 1) * 14)
                prefetch_l1(addresses.iterator + 1577 + (phase + 1) * 5)
        variant = phase_table[phase, 0]
        _rotated = (_block + workers - phase_table[phase, 2]) % workers
        _workers = workers
        if phase_table[phase, 2] < 0:
            # A recompute phase: this CTA runs every tile itself.
            _rotated = cutlass.Int32(0)
            _workers = cutlass.Int32(1)
        if _rotated < phase_table[phase, 13]:
            if variant == 0:
                _body_0(
                    work_iterators[0],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((4,), stride=(1,)), 8),),
                    None,
                    None,
                )
            if variant == 1:
                _body_1(
                    work_iterators[1],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout(1), 2),),
                    None,
                    None,
                )
            if variant == 2:
                _body_2(
                    work_iterators[2],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 4), stride=(4, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((129280, 896), stride=(896, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 896), stride=(3584, 896, 1)), 2),),
                    None,
                    None,
                )
            if variant == 3:
                _body_3(
                    work_iterators[3],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Boolean, cute.make_layout(1), 1),),
                    None,
                    None,
                )
            if variant == 4:
                _body_4(
                    work_iterators[4],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 4), stride=(4, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Boolean, cute.make_layout((1, 4), stride=(4, 1)), 1),),
                    None,
                    None,
                )
            if variant == 5:
                _body_5(
                    work_iterators[5],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1,), stride=(1,)), 8),),
                    None,
                    None,
                )
            if variant == 6:
                _body_6(
                    work_iterators[6],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout(1), 2),),
                    None,
                    None,
                )
            if variant == 7:
                _body_7(
                    work_iterators[7],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 4, 896), stride=(3584, 896, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((896,), stride=(1,)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 896), stride=(3584, 896, 1)), 2),),
                    _scratch_9,
                    None,
                )
            if variant == 8:
                _body_8(
                    work_iterators[8],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 4, 1), stride=(4, 1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((1, 4, 1), stride=(4, 1, 1)), 4),),
                    None,
                    None,
                )
            if variant == 9:
                _body_9(
                    work_iterators[9],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1, 1, 4), stride=(4, 4, 4, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Int64, cute.make_layout((1, 1, 4, 1), stride=(4, 4, 1, 1)), 8), _slot(addresses, phase_table, phase, 2, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1),),
                    None,
                    None,
                )
            if variant == 10:
                _body_10(
                    work_iterators[10],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1, 1, 1), stride=(1, 1, 1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Int64, cute.make_layout((1, 1, 1, 4), stride=(4, 4, 4, 1)), 8), _slot(addresses, phase_table, phase, 2, cutlass.Boolean, cute.make_layout((1, 4), stride=(4, 1)), 1), _slot(addresses, phase_table, phase, 3, cutlass.Boolean, cute.make_layout((1, 1, 1, 4), stride=(4, 4, 4, 1)), 1),),
                    None,
                    None,
                )
            if variant == 11:
                _body_11(
                    work_iterators[11],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout(1), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout(1), 2),),
                    None,
                    None,
                )
            if variant == 12:
                _gemm_0.run_table(
                    gemm_values[0],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    None,
                    addresses[gemm_table[phase_table[phase, 6], 4] - 1],
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                )
            if variant == 13:
                _body_13(
                    work_iterators[12],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 4, 1), stride=(4, 1, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((32,), stride=(1,)), 4), _slot(addresses, phase_table, phase, 2, cutlass.Float32, cute.make_layout((1, 4, 32), stride=(128, 32, 1)), 4),),
                    None,
                    None,
                )
            if variant == 14:
                _gemm_1.run_table(
                    gemm_values[1],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    None,
                    addresses[gemm_table[phase_table[phase, 6], 4] - 1],
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                )
            if variant == 15:
                _body_15(
                    work_iterators[13],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Boolean, cute.make_layout(1), 1), _slot(addresses, phase_table, phase, 1, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1), _slot(addresses, phase_table, phase, 2, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1),),
                    None,
                    None,
                )
            if variant == 16:
                _body_16(
                    work_iterators[14],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 4, 2, 32), stride=(128, 32, 0, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((1, 4, 2, 32), stride=(256, 64, 32, 1)), 4),),
                    None,
                    None,
                )
            if variant == 17:
                _body_17(
                    work_iterators[15],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1), _slot(addresses, phase_table, phase, 1, cutlass.Boolean, cute.make_layout((1, 1, 1, 4), stride=(4, 4, 4, 1)), 1), _slot(addresses, phase_table, phase, 2, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1),),
                    None,
                    None,
                )
            if variant == 18:
                _body_18(
                    work_iterators[16],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 4, 64), stride=(256, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 64), stride=(256, 64, 1)), 2),),
                    None,
                    None,
                )
            if variant == 19:
                _body_19(
                    work_iterators[17],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 4, 64), stride=(256, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 64), stride=(256, 64, 1)), 2),),
                    None,
                    None,
                )
            if variant == 20:
                _body_20(
                    work_iterators[18],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout(1), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout(1), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 2),),
                    None,
                    None,
                )
            if variant == 21:
                _body_21(
                    work_iterators[19],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 14, 4, 64), stride=(3584, 64, 896, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 4, 64), stride=(256, 256, 64, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 4, 64), stride=(256, 256, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 14, 4, 64), stride=(3584, 64, 896, 1)), 2),),
                    None,
                    None,
                )
            if variant == 22:
                _body_22(
                    work_iterators[20],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 2, 4, 64), stride=(512, 64, 128, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 4, 64), stride=(256, 256, 64, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 4, 64), stride=(256, 256, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 2, 4, 64), stride=(512, 64, 128, 1)), 2),),
                    None,
                    None,
                )
            if variant == 23:
                _body_23(
                    work_iterators[21],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 14, 4, 64), stride=(3584, 64, 896, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 2, 4, 64), stride=(512, 64, 128, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 2, 64), stride=(512, 128, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1), _slot(addresses, phase_table, phase, 4, cutlass.BFloat16, cute.make_layout((14, 4, 64), stride=(256, 64, 1)), 2),),
                    _scratch_26,
                    None,
                )
            if variant == 24:
                _body_24(
                    work_iterators[22],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 4, 14, 64), stride=(3584, 64, 256, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 14, 64), stride=(3584, 896, 64, 1)), 2),),
                    None,
                    None,
                )
            if variant == 25:
                _gemm_2.run_table(
                    gemm_values[2],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                )
            if variant == 26:
                _body_26(
                    work_iterators[23],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 4, 896), stride=(3584, 896, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 896), stride=(3584, 896, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 896), stride=(3584, 896, 1)), 2),),
                    None,
                    None,
                )
            if variant == 27:
                _gemm_3.run_table(
                    gemm_values[3],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                )
            if variant == 28:
                _body_28(
                    work_iterators[24],
                    _rotated,
                    _thread,
                    _workers,
                    (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 4, 4864), stride=(19456, 4864, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 4864), stride=(19456, 4864, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 4864), stride=(19456, 4864, 1)), 2),),
                    None,
                    None,
                )
            if variant == 29:
                _gemm_4.run_table(
                    gemm_values[4],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
                )
            if variant == 30:
                _gemm_5.run_table(
                    gemm_values[5],
                    _gemm_scratch,
                    table_descriptors(tma_descriptors, phase_table[phase, 6]),
                    _rotated,
                    c_address=addresses[gemm_table[phase_table[phase, 6], 3]],
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
        _gemm_4.prepare(
            gemm_tensors[4][0],
            gemm_tensors[4][1],
            gemm_tensors[4][2],
            max_active_clusters,
        ),
        _gemm_5.prepare(
            gemm_tensors[5][0],
            gemm_tensors[5][1],
            gemm_tensors[5][2],
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

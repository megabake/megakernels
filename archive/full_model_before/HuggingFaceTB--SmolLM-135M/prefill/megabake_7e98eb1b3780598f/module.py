import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm
from cutlass.cute.tensor import _Tensor

from megabake.runtime import grid_barrier
from megabake.bodies.gemm import (
    HopperWgmmaGemmPersistentKernel,
    build_table_descriptors,
    table_descriptors,
)

_gemm_0 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 64),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
)
_gemm_1 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 64),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
)
_gemm_2 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 64),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
)
_gemm_3 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 64),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
)
_gemm_4 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 64),
    (1, 1),
    1,
    True,
    smem_capacity=148480,
    reallocate_registers=False,
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
    address = addresses[phase_table[phase, 4 + column]]
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
    epoch: cutlass.Int32,
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    work_iterators: tuple[cute.Tensor, ...],
    dynamic_addresses: cutlass.Int64,
    dynamic_slot_table: cute.Tensor,
    gemm_table: cute.Tensor,
    tma_descriptors: cute.Tensor,
    gemm_values,
):
    _, _, _block = cute.arch.block_idx()
    _thread, _, _ = cute.arch.thread_idx()
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o0 = linear_index % 4
                value = (o0 + cutlass.Int64(0))
                output[o0] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                value = cutlass.BFloat16(0.0)
                output[0] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 576
                linear_index = linear_index // 576
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1] if 0 <= (0) < 1 and 0 <= (o1) < 4 else cutlass.Int64(0))
                if load_0 < 0:
                    llvm.intr_trap()
                if load_0 >= 49152:
                    llvm.intr_trap()
                load_1 = (tensor_bindings[1][(tensor_bindings[0][0, o1] if 0 <= (0) < 1 and 0 <= (o1) < 4 else cutlass.Int64(0)), o2] if 0 <= (o2) < 576 else cutlass.BFloat16(0))
                value = load_1
                output[o0, o1, o2] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                value = cutlass.Boolean(True)
                output[0] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1] if 0 <= (0) < 1 and 0 <= (o1) < 4 else cutlass.Int64(0))
                value = (load_0).to(cutlass.Boolean)
                output[o0, o1] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o0 = linear_index % 1
                value = o0
                output[o0] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                value = cutlass.BFloat16(-3.3895313892515355e+38)
                output[0] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            row_index = work_iterator[tile]
            if row_index < 4:
                o1 = (row_index // 1) % 4
                o0 = (row_index // 4) % 1
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for reduction_index in cutlass.range(thread, 576, 256):
                    r0 = reduction_index
                    accumulator = (accumulator + cute.pow(((tensor_bindings[0][o0, o1, r0] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (r0) < 576 else cutlass.BFloat16(0))).to(cutlass.Float32), (cutlass.Int64(2)).to(cutlass.Float32)))
                warp_partial = cute.arch.warp_reduction_sum(accumulator)
                if lane == 0:
                    scratch[ warp] = warp_partial
                cute.arch.sync_threads()
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for warp_index in cutlass.range(warps):
                    accumulator = accumulator + scratch[ warp_index]
                for post_index in cutlass.range(thread, 576, 256):
                    o2 = (post_index // 1) % 576
                    post_load_0 = (tensor_bindings[0][o0, o1, o2] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 576 else cutlass.BFloat16(0))
                    post_load_1 = (tensor_bindings[1][o2] if 0 <= (o2) < 576 else cutlass.BFloat16(0))
                    output[o0, o1, o2] = (((post_load_1).to(cutlass.Float32) * ((((post_load_0).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(576)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o2 = linear_index % 1
                linear_index = linear_index // 1
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, 0] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
                value = (load_0).to(cutlass.Float32)
                output[o0, o1, o2] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 4
                linear_index = linear_index // 4
                o2 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, 0, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3) < 4 else cutlass.Int64(0))
                load_1 = (tensor_bindings[1][0, 0, o2, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
                value = (load_0 <= load_1)
                output[o0, o1, o2, o3] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                linear_index = linear_index // 1
                o2 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
                load_1 = (tensor_bindings[1][0, 0, 0, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3) < 4 else cutlass.Int64(0))
                if load_0 < 0:
                    llvm.intr_trap()
                if load_0 >= 1:
                    llvm.intr_trap()
                if load_1 < 0:
                    llvm.intr_trap()
                if load_1 >= 4:
                    llvm.intr_trap()
                load_2 = tensor_bindings[2][(tensor_bindings[0][0, 0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), (tensor_bindings[1][0, 0, 0, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3) < 4 else cutlass.Int64(0))]
                value = load_2
                output[o0, o1, o2, o3] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                load_0 = tensor_bindings[0][0]
                value = load_0
                output[0] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 32
                linear_index = linear_index // 32
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, 0] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
                load_1 = (tensor_bindings[1][o2] if 0 <= (o2) < 32 else cutlass.Float32(0))
                value = (load_0 * load_1)
                output[o0, o1, o2] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 4
                linear_index = linear_index // 4
                o2 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = tensor_bindings[0][0]
                load_1 = (tensor_bindings[1][0, 0, o2, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 4 and 0 <= (o3) < 4 else cutlass.Boolean(0))
                value = (load_0 & load_1)
                output[o0, o1, o2, o3] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 32
                linear_index = linear_index // 32
                o2 = linear_index % 2
                linear_index = linear_index // 2
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, o2, o3] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 2 and 0 <= (o3) < 32 else cutlass.Float32(0))
                value = load_0
                output[o0, o1, o2, o3] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 4
                linear_index = linear_index // 4
                o2 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, o2, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 4 and 0 <= (o3) < 4 else cutlass.Boolean(0))
                load_1 = (tensor_bindings[1][0, 0, 0, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3) < 4 else cutlass.Boolean(0))
                value = (load_0 & load_1)
                output[o0, o1, o2, o3] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 64
                linear_index = linear_index // 64
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, o2] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 64 else cutlass.Float32(0))
                value = ((cute.cos(load_0) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
                output[o0, o1, o2] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 64
                linear_index = linear_index // 64
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, o2] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 64 else cutlass.Float32(0))
                value = ((cute.sin(load_0) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
                output[o0, o1, o2] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 4
                linear_index = linear_index // 4
                o2 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, o2, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 4 and 0 <= (o3) < 4 else cutlass.Boolean(0))
                load_1 = tensor_bindings[1][0]
                load_2 = tensor_bindings[2][0]
                value = ((load_1) if (load_0) else (load_2))
                output[o0, o1, o2, o3] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 64
                linear_index = linear_index // 64
                o1 = linear_index % 9
                linear_index = linear_index // 9
                o2 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, o2, (32 + o3)] if 0 <= (0) < 1 and 0 <= (o1) < 9 and 0 <= (o2) < 4 and 0 <= ((32 + o3)) < 64 else cutlass.BFloat16(0))
                load_1 = (tensor_bindings[0][0, o1, o2, (-32 + o3)] if 0 <= (0) < 1 and 0 <= (o1) < 9 and 0 <= (o2) < 4 and 0 <= ((-32 + o3)) < 64 else cutlass.BFloat16(0))
                load_2 = (tensor_bindings[1][0, 0, o2, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 4 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_3 = (tensor_bindings[0][0, o1, o2, o3] if 0 <= (0) < 1 and 0 <= (o1) < 9 and 0 <= (o2) < 4 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_4 = (tensor_bindings[2][0, 0, o2, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 4 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                value = ((((((load_3).to(cutlass.Float32) * (load_4).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0))) if ((o3 < cutlass.Int64(32))) else (load_1))).to(cutlass.Float32) * (load_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2, o3] = value

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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 64
                linear_index = linear_index // 64
                o1 = linear_index % 3
                linear_index = linear_index // 3
                o2 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, o2, (32 + o3)] if 0 <= (0) < 1 and 0 <= (o1) < 3 and 0 <= (o2) < 4 and 0 <= ((32 + o3)) < 64 else cutlass.BFloat16(0))
                load_1 = (tensor_bindings[0][0, o1, o2, (-32 + o3)] if 0 <= (0) < 1 and 0 <= (o1) < 3 and 0 <= (o2) < 4 and 0 <= ((-32 + o3)) < 64 else cutlass.BFloat16(0))
                load_2 = (tensor_bindings[1][0, 0, o2, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 4 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_3 = (tensor_bindings[0][0, o1, o2, o3] if 0 <= (0) < 1 and 0 <= (o1) < 3 and 0 <= (o2) < 4 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_4 = (tensor_bindings[2][0, 0, o2, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 4 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                value = ((((((load_3).to(cutlass.Float32) * (load_4).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0))) if ((o3 < cutlass.Int64(32))) else (load_1))).to(cutlass.Float32) * (load_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2, o3] = value

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
        row_count = 36
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            row = work_iterator[tile] * 8 + warp
            if warp < 8:
                if row < row_count:
                    bh = row // 4
                    s = row % 4
                    b = bh // 9
                    h = bh % 9
                    kv_head = h // 3
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 64
                linear_index = linear_index // 64
                o2 = linear_index % 9
                linear_index = linear_index // 9
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, o2, o3] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 9 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                value = load_0
                output[o0, o1, o2, o3] = value

    @cute.jit
    def _body_25(
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
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 576
                linear_index = linear_index // 576
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, o2] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 576 else cutlass.BFloat16(0))
                load_1 = (tensor_bindings[1][0, o1, o2] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 576 else cutlass.BFloat16(0))
                value = (((load_0).to(cutlass.Float32) + (load_1).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2] = value

    @cute.jit
    def _body_27(
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
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 1536
                linear_index = linear_index // 1536
                o1 = linear_index % 4
                linear_index = linear_index // 4
                o0 = linear_index % 1
                load_0 = (tensor_bindings[0][0, o1, o2] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 1536 else cutlass.BFloat16(0))
                load_1 = (tensor_bindings[1][0, o1, o2] if 0 <= (0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 1536 else cutlass.BFloat16(0))
                value = ((((((load_0).to(cutlass.Float32) / (cute.exp((-((load_0).to(cutlass.Float32)))) + (cutlass.Int64(1)).to(cutlass.Float32)))).to(cutlass.BFloat16)).to(cutlass.Float32) * (load_1).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2] = value


    # Publish this call's boundary addresses before any phase reads them.
    if _block == 0:
        _dynamic = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int64, dynamic_addresses, cute.AddressSpace.gmem, assumed_align=8
            ),
            cute.make_layout(1006),
        )
        for _position in cutlass.range(_thread, 1006, 256, unroll=1):
            addresses[dynamic_slot_table[_position]] = _dynamic[_position]

    grid_barrier(ready, workers, epoch)
    _warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if _warp == 0:
        for gemm in cutlass.range(_block, 211, workers, unroll=1):
            gemm_variant = gemm_table[gemm, 0]
            if gemm_variant == 0:
                build_table_descriptors(
                    gemm_values[0],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (4, 576, 576),
                )
            if gemm_variant == 1:
                build_table_descriptors(
                    gemm_values[1],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (4, 192, 576),
                )
            if gemm_variant == 2:
                build_table_descriptors(
                    gemm_values[2],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (4, 1536, 576),
                )
            if gemm_variant == 3:
                build_table_descriptors(
                    gemm_values[3],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (4, 576, 1536),
                )
            if gemm_variant == 4:
                build_table_descriptors(
                    gemm_values[4],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (4, 49152, 576),
                )
        cute.nvgpu.cpasync.fence_tma_desc_release()
    grid_barrier(ready, workers, cutlass.Int32(1) + epoch)

    for phase in cutlass.range(502, unroll=1):
        variant = phase_table[phase, 0]
        _rotated = (_block + workers - phase_table[phase, 2]) % workers
        if variant == 0:
            _body_0(
                work_iterators[0],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((4,), stride=(1,)), 8),),
                None,
                None,
            )
        if variant == 1:
            _body_1(
                work_iterators[1],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout(1), 2),),
                None,
                None,
            )
        if variant == 2:
            _body_2(
                work_iterators[2],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 4), stride=(4, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((49152, 576), stride=(576, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 576), stride=(2304, 576, 1)), 2),),
                None,
                None,
            )
        if variant == 3:
            _body_3(
                work_iterators[3],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Boolean, cute.make_layout(1), 1),),
                None,
                None,
            )
        if variant == 4:
            _body_4(
                work_iterators[4],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 4), stride=(4, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Boolean, cute.make_layout((1, 4), stride=(4, 1)), 1),),
                None,
                None,
            )
        if variant == 5:
            _body_5(
                work_iterators[5],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1,), stride=(1,)), 8),),
                None,
                None,
            )
        if variant == 6:
            _body_6(
                work_iterators[6],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout(1), 2),),
                None,
                None,
            )
        if variant == 7:
            _body_7(
                work_iterators[7],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 4, 576), stride=(2304, 576, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((576,), stride=(1,)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 576), stride=(2304, 576, 1)), 2),),
                _scratch_9,
                None,
            )
        if variant == 8:
            _body_8(
                work_iterators[8],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 4, 1), stride=(4, 1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((1, 4, 1), stride=(4, 1, 1)), 4),),
                None,
                None,
            )
        if variant == 9:
            _body_9(
                work_iterators[9],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1, 1, 4), stride=(4, 4, 4, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Int64, cute.make_layout((1, 1, 4, 1), stride=(4, 4, 1, 1)), 8), _slot(addresses, phase_table, phase, 2, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1),),
                None,
                None,
            )
        if variant == 10:
            _body_10(
                work_iterators[10],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1, 1, 1), stride=(1, 1, 1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Int64, cute.make_layout((1, 1, 1, 4), stride=(4, 4, 4, 1)), 8), _slot(addresses, phase_table, phase, 2, cutlass.Boolean, cute.make_layout((1, 4), stride=(4, 1)), 1), _slot(addresses, phase_table, phase, 3, cutlass.Boolean, cute.make_layout((1, 1, 1, 4), stride=(4, 4, 4, 1)), 1),),
                None,
                None,
            )
        if variant == 11:
            _body_11(
                work_iterators[11],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout(1), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout(1), 2),),
                None,
                None,
            )
        if variant == 12:
            _gemm_0.run_table(
                gemm_values[0],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if variant == 13:
            _body_13(
                work_iterators[12],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 4, 1), stride=(4, 1, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((32,), stride=(1,)), 4), _slot(addresses, phase_table, phase, 2, cutlass.Float32, cute.make_layout((1, 4, 32), stride=(128, 32, 1)), 4),),
                None,
                None,
            )
        if variant == 14:
            _gemm_1.run_table(
                gemm_values[1],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if variant == 15:
            _body_15(
                work_iterators[13],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Boolean, cute.make_layout(1), 1), _slot(addresses, phase_table, phase, 1, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1), _slot(addresses, phase_table, phase, 2, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1),),
                None,
                None,
            )
        if variant == 16:
            _body_16(
                work_iterators[14],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 4, 2, 32), stride=(128, 32, 0, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((1, 4, 2, 32), stride=(256, 64, 32, 1)), 4),),
                None,
                None,
            )
        if variant == 17:
            _body_17(
                work_iterators[15],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1), _slot(addresses, phase_table, phase, 1, cutlass.Boolean, cute.make_layout((1, 1, 1, 4), stride=(4, 4, 4, 1)), 1), _slot(addresses, phase_table, phase, 2, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1),),
                None,
                None,
            )
        if variant == 18:
            _body_18(
                work_iterators[16],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 4, 64), stride=(256, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 64), stride=(256, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 19:
            _body_19(
                work_iterators[17],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 4, 64), stride=(256, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 64), stride=(256, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 20:
            _body_20(
                work_iterators[18],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout(1), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout(1), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 2),),
                None,
                None,
            )
        if variant == 21:
            _body_21(
                work_iterators[19],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 9, 4, 64), stride=(2304, 64, 576, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 4, 64), stride=(256, 256, 64, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 4, 64), stride=(256, 256, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 9, 4, 64), stride=(2304, 64, 576, 1)), 2),),
                None,
                None,
            )
        if variant == 22:
            _body_22(
                work_iterators[20],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 3, 4, 64), stride=(768, 64, 192, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 4, 64), stride=(256, 256, 64, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 4, 64), stride=(256, 256, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 3, 4, 64), stride=(768, 64, 192, 1)), 2),),
                None,
                None,
            )
        if variant == 23:
            _body_23(
                work_iterators[21],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 9, 4, 64), stride=(2304, 64, 576, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 3, 4, 64), stride=(768, 64, 192, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 3, 64), stride=(768, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.Boolean, cute.make_layout((1, 1, 4, 4), stride=(16, 16, 4, 1)), 1), _slot(addresses, phase_table, phase, 4, cutlass.BFloat16, cute.make_layout((9, 4, 64), stride=(256, 64, 1)), 2),),
                _scratch_26,
                None,
            )
        if variant == 24:
            _body_24(
                work_iterators[22],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 4, 9, 64), stride=(2304, 64, 256, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 9, 64), stride=(2304, 576, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 25:
            _body_25(
                work_iterators[23],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 4, 576), stride=(2304, 576, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 576), stride=(2304, 576, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 576), stride=(2304, 576, 1)), 2),),
                None,
                None,
            )
        if variant == 26:
            _gemm_2.run_table(
                gemm_values[2],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if variant == 27:
            _body_27(
                work_iterators[24],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 4, 1536), stride=(6144, 1536, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 4, 1536), stride=(6144, 1536, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 4, 1536), stride=(6144, 1536, 1)), 2),),
                None,
                None,
            )
        if variant == 28:
            _gemm_3.run_table(
                gemm_values[3],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if variant == 29:
            _gemm_4.run_table(
                gemm_values[4],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if phase_table[phase, 1] != 0:
            grid_barrier(ready, workers, epoch + phase + 2)
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
    epoch: cutlass.Int32,
    addresses: cute.Tensor,
    phase_table: cute.Tensor,
    work_iterators: tuple[cute.Tensor, ...],
    dynamic_addresses: cutlass.Int64,
    dynamic_slot_table: cute.Tensor,
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
    )
    megabake_kernel(
        ready,
        workers,
        epoch,
        addresses,
        phase_table,
        work_iterators,
        dynamic_addresses,
        dynamic_slot_table,
        gemm_table,
        tma_descriptors,
        gemm_values,
    ).launch(
        grid=[1, 1, workers],
        block=[256, 1, 1],
        cooperative=True,
        stream=stream,
    )

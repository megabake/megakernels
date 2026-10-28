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
class _Phase31Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase157Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 2048],
        16
    ]


@cute.union
class _WorkerScratch:
    phase_31: _Phase31Scratch
    phase_157: _Phase157Scratch

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
    _scratch_31 = _worker_scratch.phase_31.body.get_tensor(cute.make_layout(8))
    _scratch_157 = _worker_scratch.phase_157.body.get_tensor(cute.make_layout(2048))
    if _thread == 0:
        _worker_scratch.phase_31.barrier_0[0] = 0
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
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 576
                linear_index = linear_index // 576
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
                if load_0 < 0:
                    llvm.intr_trap()
                if load_0 >= 49152:
                    llvm.intr_trap()
                load_1 = (tensor_bindings[1][(tensor_bindings[0][0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0)), o2] if 0 <= (o2) < 576 else cutlass.BFloat16(0))
                value = load_1
                output[o0, o1, o2] = value

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
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                linear_index = linear_index // 1
                o2 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Int64(0))
                value = (load_0).to(cutlass.Float32)
                output[o0, o1, o2] = value

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
                for reduction_index in cutlass.range(thread, 576, 256):
                    r0 = reduction_index
                    accumulator = (accumulator + cute.pow(((tensor_bindings[0][o0, o1, r0] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (r0) < 576 else cutlass.BFloat16(0))).to(cutlass.Float32), (cutlass.Int64(2)).to(cutlass.Float32)))
                warp_partial = cute.arch.warp_reduction_sum(accumulator)
                if lane == 0:
                    scratch[ warp] = warp_partial
                cute.arch.sync_threads()
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for warp_index in cutlass.range(warps):
                    accumulator = accumulator + scratch[ warp_index]
                for post_index in cutlass.range(thread, 576, 256):
                    o2 = (post_index // 1) % 576
                    post_load_0 = (tensor_bindings[0][o0, o1, o2] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 576 else cutlass.BFloat16(0))
                    post_load_1 = (tensor_bindings[1][o2] if 0 <= (o2) < 576 else cutlass.BFloat16(0))
                    output[o0, o1, o2] = (((post_load_1).to(cutlass.Float32) * ((((post_load_0).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(576)).to(cutlass.Float32)) + cutlass.Float32(1e-05))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
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
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 32
                linear_index = linear_index // 32
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 else cutlass.Float32(0))
                load_1 = (tensor_bindings[1][o2] if 0 <= (o2) < 32 else cutlass.Float32(0))
                value = (load_0 * load_1)
                output[o0, o1, o2] = value

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
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o3 = linear_index % 32
                linear_index = linear_index // 32
                o2 = linear_index % 2
                linear_index = linear_index // 2
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, o2, o3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 2 and 0 <= (o3) < 32 else cutlass.Float32(0))
                value = load_0
                output[o0, o1, o2, o3] = value

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
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 64
                linear_index = linear_index // 64
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, o2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 64 else cutlass.Float32(0))
                value = ((cute.cos(load_0) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
                output[o0, o1, o2] = value

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
                o2 = linear_index % 64
                linear_index = linear_index // 64
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, o2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 64 else cutlass.Float32(0))
                value = ((cute.sin(load_0) * cutlass.Float32(1.0))).to(cutlass.BFloat16)
                output[o0, o1, o2] = value

    @cute.jit
    def _body_9(
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
                o2 = linear_index % 9
                linear_index = linear_index // 9
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][o0, o1, o2, o3] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 9 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_1 = (tensor_bindings[1][o0, o1, o2, o3] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 9 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_2 = (tensor_bindings[0][o0, o1, o2, (32 + o3)] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 9 and 0 <= ((32 + o3)) < 64 else cutlass.BFloat16(0))
                load_3 = (tensor_bindings[2][o0, o1, o2, o3] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 9 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_4 = (tensor_bindings[2][o0, o1, o2, (-32 + o3)] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 9 and 0 <= ((-32 + o3)) < 64 else cutlass.BFloat16(0))
                load_5 = (tensor_bindings[0][o0, o1, o2, (-32 + o3)] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 9 and 0 <= ((-32 + o3)) < 64 else cutlass.BFloat16(0))
                load_6 = (tensor_bindings[1][o0, o1, o2, (-32 + o3)] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 9 and 0 <= ((-32 + o3)) < 64 else cutlass.BFloat16(0))
                value = ((((((((load_0).to(cutlass.Float32) * (load_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) - ((((load_2).to(cutlass.Float32) * (load_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3 < cutlass.Int64(32))) else (((((((((load_0).to(cutlass.Float32) * (load_4).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + ((((load_5).to(cutlass.Float32) * (load_6).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3 < cutlass.Int64(64))) else (load_0))))
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
                o3 = linear_index % 64
                linear_index = linear_index // 64
                o2 = linear_index % 3
                linear_index = linear_index // 3
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][o0, o1, o2, o3] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 3 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_1 = (tensor_bindings[1][o0, o1, o2, o3] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 3 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_2 = (tensor_bindings[0][o0, o1, o2, (32 + o3)] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 3 and 0 <= ((32 + o3)) < 64 else cutlass.BFloat16(0))
                load_3 = (tensor_bindings[2][o0, o1, o2, o3] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 3 and 0 <= (o3) < 64 else cutlass.BFloat16(0))
                load_4 = (tensor_bindings[2][o0, o1, o2, (-32 + o3)] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 3 and 0 <= ((-32 + o3)) < 64 else cutlass.BFloat16(0))
                load_5 = (tensor_bindings[0][o0, o1, o2, (-32 + o3)] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 3 and 0 <= ((-32 + o3)) < 64 else cutlass.BFloat16(0))
                load_6 = (tensor_bindings[1][o0, o1, o2, (-32 + o3)] if 0 <= (o0) < 1 and 0 <= (o1) < 1 and 0 <= (o2) < 3 and 0 <= ((-32 + o3)) < 64 else cutlass.BFloat16(0))
                value = ((((((((load_0).to(cutlass.Float32) * (load_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) - ((((load_2).to(cutlass.Float32) * (load_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3 < cutlass.Int64(32))) else (((((((((load_0).to(cutlass.Float32) * (load_4).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + ((((load_5).to(cutlass.Float32) * (load_6).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)) if ((o3 < cutlass.Int64(64))) else (load_0))))
                output[o0, o1, o2, o3] = value

    @cute.jit
    def _body_11(
            work_iterator: cute.Tensor,
            block: cutlass.Int32,
            thread: cutlass.Int32,
            workers: cutlass.Int32,
        tensor_bindings: tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        scratch,
        epilogue,
    ):
        row_count = 9
        current_length = tensor_bindings[5][0].to(cutlass.Int32)
        if current_length < 0:
            llvm.intr_trap()
        if current_length + cutlass.Int32(1) > cutlass.Int32(128):
            llvm.intr_trap()
        # Each of the first 8 warps takes whole query rows: lanes split the
        # visible keys for scores and the head dimensions for the value sum, and
        # the row's scores and probabilities stay in that warp's scratch slice.
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            warp = thread // 32
            lane = thread % 32
            if warp < 8:
                for bh in cutlass.range(cutlass.Int32(work_iterator[tile]) * 8 + warp, row_count, 72):
                    s = 0
                    b = bh // 9
                    h = bh % 9
                    kv_head = h // 3
                    base = warp * 128
                    maximum = cutlass.Float32(-1.0e30)
                    for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                        score = cutlass.Float32(0.0)
                        if key < current_length:
                            for d in cutlass.range_constexpr(64):
                                score = score + tensor_bindings[0][b, s, h, d].to(cutlass.Float32) * tensor_bindings[3][b, key, kv_head, d].to(cutlass.Float32)
                        else:
                            for d in cutlass.range_constexpr(64):
                                score = score + tensor_bindings[0][b, s, h, d].to(cutlass.Float32) * tensor_bindings[1][b, 0, kv_head, d].to(cutlass.Float32)
                        score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                        score = score * cutlass.Float32(0.125)
                        scratch[base + key] = score
                        maximum = cute.arch.fmax(maximum, score, nan=True)
                    maximum = cute.arch.warp_reduction_max(maximum)
                    total = cutlass.Float32(0.0)
                    for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                        probability = cute.exp(scratch[base + key] - maximum)
                        scratch[base + key] = probability
                        total = total + probability
                    total = cute.arch.warp_reduction_sum(total)
                    denominator = total
                    if total <= cutlass.Float32(0.0):
                        denominator = cutlass.Float32(1.0)
                    for key in cutlass.range(lane, current_length + cutlass.Int32(1), 32):
                        scratch[base + key] = (scratch[base + key] / denominator).to(cutlass.BFloat16).to(cutlass.Float32)
                    cute.arch.sync_warp()
                    for d in cutlass.range(lane, 64, 32):
                        value = cutlass.Float32(0.0)
                        # Cached keys first, then the new key: the source key order.
                        # Eight loads are issued before they are accumulated, so
                        # their latencies overlap; the sum order is unchanged.
                        blocked = current_length - current_length % 8
                        for block_start in cutlass.range(0, blocked, 8):
                            value_0 = tensor_bindings[4][b, block_start + 0, kv_head, d]
                            value_1 = tensor_bindings[4][b, block_start + 1, kv_head, d]
                            value_2 = tensor_bindings[4][b, block_start + 2, kv_head, d]
                            value_3 = tensor_bindings[4][b, block_start + 3, kv_head, d]
                            value_4 = tensor_bindings[4][b, block_start + 4, kv_head, d]
                            value_5 = tensor_bindings[4][b, block_start + 5, kv_head, d]
                            value_6 = tensor_bindings[4][b, block_start + 6, kv_head, d]
                            value_7 = tensor_bindings[4][b, block_start + 7, kv_head, d]
                            value = value + scratch[base + block_start + 0] * value_0.to(cutlass.Float32)
                            value = value + scratch[base + block_start + 1] * value_1.to(cutlass.Float32)
                            value = value + scratch[base + block_start + 2] * value_2.to(cutlass.Float32)
                            value = value + scratch[base + block_start + 3] * value_3.to(cutlass.Float32)
                            value = value + scratch[base + block_start + 4] * value_4.to(cutlass.Float32)
                            value = value + scratch[base + block_start + 5] * value_5.to(cutlass.Float32)
                            value = value + scratch[base + block_start + 6] * value_6.to(cutlass.Float32)
                            value = value + scratch[base + block_start + 7] * value_7.to(cutlass.Float32)
                        for key in cutlass.range(blocked, current_length):
                            value = value + scratch[base + key] * tensor_bindings[4][b, key, kv_head, d].to(cutlass.Float32)
                        value = value + scratch[base + current_length] * tensor_bindings[2][b, 0, kv_head, d].to(cutlass.Float32)
                        tensor_bindings[6][bh, s, d] = value.to(cutlass.BFloat16)
                    # Keys before current_length come from the cache and the new
                    # key from new_k, so appending here cannot race with readers.
                    if h % 3 == 0:
                        for d in cutlass.range(lane, 64, 32):
                            tensor_bindings[3][b, current_length, kv_head, d] = tensor_bindings[1][b, 0, kv_head, d]
                            tensor_bindings[4][b, current_length, kv_head, d] = tensor_bindings[2][b, 0, kv_head, d]
                    cute.arch.sync_warp()

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
            linear_index = work_iterator[tile] + thread
            if linear_index < output_count:
                o2 = linear_index % 576
                linear_index = linear_index // 576
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, o2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 576 else cutlass.BFloat16(0))
                load_1 = (tensor_bindings[1][0, 0, o2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 576 else cutlass.BFloat16(0))
                value = (((load_0).to(cutlass.Float32) + (load_1).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2] = value

    @cute.jit
    def _body_14(
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
                o0 = linear_index % 1
                linear_index = linear_index // 1
                o1 = linear_index % 1
                load_0 = (tensor_bindings[0][0, 0, o2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 1536 else cutlass.BFloat16(0))
                load_1 = (tensor_bindings[1][0, 0, o2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2) < 1536 else cutlass.BFloat16(0))
                value = ((((((load_0).to(cutlass.Float32) / (cute.exp((-((load_0).to(cutlass.Float32)))) + (cutlass.Int64(1)).to(cutlass.Float32)))).to(cutlass.BFloat16)).to(cutlass.Float32) * (load_1).to(cutlass.Float32))).to(cutlass.BFloat16)
                output[o0, o1, o2] = value


    # Publish this call's boundary addresses before any phase reads them.
    if _block == 0:
        _dynamic = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int64, dynamic_addresses, cute.AddressSpace.gmem, assumed_align=8
            ),
            cute.make_layout(1209),
        )
        for _position in cutlass.range(_thread, 1209, 256, unroll=1):
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
                    (1, 576, 576),
                )
            if gemm_variant == 1:
                build_table_descriptors(
                    gemm_values[1],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (1, 192, 576),
                )
            if gemm_variant == 2:
                build_table_descriptors(
                    gemm_values[2],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (1, 1536, 576),
                )
            if gemm_variant == 3:
                build_table_descriptors(
                    gemm_values[3],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (1, 576, 1536),
                )
            if gemm_variant == 4:
                build_table_descriptors(
                    gemm_values[4],
                    tma_descriptors,
                    gemm,
                    addresses[gemm_table[gemm, 1]],
                    addresses[gemm_table[gemm, 2]],
                    addresses[gemm_table[gemm, 3]],
                    (1, 49152, 576),
                )
        cute.nvgpu.cpasync.fence_tma_desc_release()
    grid_barrier(ready, workers, cutlass.Int32(1) + epoch)

    for phase in cutlass.range(603, unroll=1):
        variant = phase_table[phase, 0]
        _rotated = (_block + workers - phase_table[phase, 2]) % workers
        if variant == 0:
            _body_0(
                work_iterators[0],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1), stride=(1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((49152, 576), stride=(576, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2),),
                None,
                None,
            )
        if variant == 1:
            _body_1(
                work_iterators[1],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Int64, cute.make_layout((1, 1, 1), stride=(1, 1, 1)), 8), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((1, 1, 1), stride=(1, 1, 1)), 4),),
                None,
                None,
            )
        if variant == 2:
            _body_2(
                work_iterators[2],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((576,), stride=(1,)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2),),
                _scratch_31,
                None,
            )
        if variant == 3:
            _body_3(
                work_iterators[3],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 1), stride=(1, 1, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((32,), stride=(1,)), 4), _slot(addresses, phase_table, phase, 2, cutlass.Float32, cute.make_layout((1, 1, 32), stride=(32, 32, 1)), 4),),
                None,
                None,
            )
        if variant == 4:
            _gemm_0.run_table(
                gemm_values[0],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if variant == 5:
            _body_5(
                work_iterators[4],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 2, 32), stride=(32, 32, 0, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.Float32, cute.make_layout((1, 1, 2, 32), stride=(64, 64, 32, 1)), 4),),
                None,
                None,
            )
        if variant == 6:
            _gemm_1.run_table(
                gemm_values[1],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if variant == 7:
            _body_7(
                work_iterators[5],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 8:
            _body_8(
                work_iterators[6],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.Float32, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 4), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 64), stride=(64, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 9:
            _body_9(
                work_iterators[7],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 9, 64), stride=(576, 576, 64, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 9, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 9, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 1, 9, 64), stride=(576, 576, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 10:
            _body_10(
                work_iterators[8],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(192, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(64, 64, 0, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(192, 192, 64, 1)), 2),),
                None,
                None,
            )
        if variant == 11:
            _body_11(
                work_iterators[9],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 9, 64), stride=(576, 576, 64, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(192, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 3, 64), stride=(192, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 3, cutlass.BFloat16, cute.make_layout((1, 128, 3, 64), stride=(24576, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 4, cutlass.BFloat16, cute.make_layout((1, 128, 3, 64), stride=(24576, 192, 64, 1)), 2), _slot(addresses, phase_table, phase, 5, cutlass.Int32, cute.make_layout((1,), stride=(1,)), 4), _slot(addresses, phase_table, phase, 6, cutlass.BFloat16, cute.make_layout((9, 1, 64), stride=(64, 64, 1)), 2),),
                _scratch_157,
                None,
            )
        if variant == 12:
            _body_12(
                work_iterators[10],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 576), stride=(576, 576, 1)), 2),),
                None,
                None,
            )
        if variant == 13:
            _gemm_2.run_table(
                gemm_values[2],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if variant == 14:
            _body_14(
                work_iterators[11],
                _rotated,
                _thread,
                workers,
                (_slot(addresses, phase_table, phase, 0, cutlass.BFloat16, cute.make_layout((1, 1, 1536), stride=(1536, 1536, 1)), 2), _slot(addresses, phase_table, phase, 1, cutlass.BFloat16, cute.make_layout((1, 1, 1536), stride=(1536, 1536, 1)), 2), _slot(addresses, phase_table, phase, 2, cutlass.BFloat16, cute.make_layout((1, 1, 1536), stride=(1536, 1536, 1)), 2),),
                None,
                None,
            )
        if variant == 15:
            _gemm_3.run_table(
                gemm_values[3],
                _gemm_scratch,
                table_descriptors(tma_descriptors, phase_table[phase, 4]),
                _rotated,
            )
        if variant == 16:
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

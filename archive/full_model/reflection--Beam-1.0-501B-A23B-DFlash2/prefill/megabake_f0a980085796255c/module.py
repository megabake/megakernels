import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass._mlir.dialects import llvm

from megabake.runtime import cooperative_worker_loop, grid_barrier
from megabake.bodies.gemm import HopperWgmmaGemmPersistentKernel

_g0_body_13 = HopperWgmmaGemmPersistentKernel(
    cutlass.Float32,
    (64, 64),
    (1, 1),
    1,
    True,
)


@cute.struct
class _Phase10Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase21Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase23Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase34Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 64],
        16
    ]

@cute.struct
class _Phase38Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase44Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase48Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase49Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase52Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 64],
        16
    ]

@cute.struct
class _Phase56Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase62Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase66Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase67Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase70Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 64],
        16
    ]

@cute.struct
class _Phase74Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase80Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase84Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase85Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase88Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 64],
        16
    ]

@cute.struct
class _Phase92Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase98Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase102Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase103Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase106Scratch:
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 64],
        16
    ]

@cute.struct
class _Phase110Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]

@cute.struct
class _Phase116Scratch:
    barrier_0: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 1], 8
    ]
    body: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, 8],
        16
    ]


@cute.union
class _WorkerScratch:
    phase_10: _Phase10Scratch
    phase_21: _Phase21Scratch
    phase_23: _Phase23Scratch
    phase_34: _Phase34Scratch
    phase_38: _Phase38Scratch
    phase_44: _Phase44Scratch
    phase_48: _Phase48Scratch
    phase_49: _Phase49Scratch
    phase_52: _Phase52Scratch
    phase_56: _Phase56Scratch
    phase_62: _Phase62Scratch
    phase_66: _Phase66Scratch
    phase_67: _Phase67Scratch
    phase_70: _Phase70Scratch
    phase_74: _Phase74Scratch
    phase_80: _Phase80Scratch
    phase_84: _Phase84Scratch
    phase_85: _Phase85Scratch
    phase_88: _Phase88Scratch
    phase_92: _Phase92Scratch
    phase_98: _Phase98Scratch
    phase_102: _Phase102Scratch
    phase_103: _Phase103Scratch
    phase_106: _Phase106Scratch
    phase_110: _Phase110Scratch
    phase_116: _Phase116Scratch

@cute.kernel
def megabake_kernel(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    epoch: cutlass.Int32,
    work_iterators: tuple[cute.Tensor, ...],
    tensors: tuple[cute.Tensor, ...],
    g0_values: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...]
):
    _, _, _block = cute.arch.block_idx()
    _thread, _, _ = cute.arch.thread_idx()
    _worker_scratch = cutlass.utils.SmemAllocator().allocate(_WorkerScratch)
    _scratch_10 = _worker_scratch.phase_10.body.get_tensor(cute.make_layout(8))
    _scratch_21 = _worker_scratch.phase_21.body.get_tensor(cute.make_layout(8))
    _scratch_23 = _worker_scratch.phase_23.body.get_tensor(cute.make_layout(8))
    _scratch_34 = _worker_scratch.phase_34.body.get_tensor(cute.make_layout(64))
    _scratch_38 = _worker_scratch.phase_38.body.get_tensor(cute.make_layout(8))
    _scratch_44 = _worker_scratch.phase_44.body.get_tensor(cute.make_layout(8))
    _scratch_48 = _worker_scratch.phase_48.body.get_tensor(cute.make_layout(8))
    _scratch_49 = _worker_scratch.phase_49.body.get_tensor(cute.make_layout(8))
    _scratch_52 = _worker_scratch.phase_52.body.get_tensor(cute.make_layout(64))
    _scratch_56 = _worker_scratch.phase_56.body.get_tensor(cute.make_layout(8))
    _scratch_62 = _worker_scratch.phase_62.body.get_tensor(cute.make_layout(8))
    _scratch_66 = _worker_scratch.phase_66.body.get_tensor(cute.make_layout(8))
    _scratch_67 = _worker_scratch.phase_67.body.get_tensor(cute.make_layout(8))
    _scratch_70 = _worker_scratch.phase_70.body.get_tensor(cute.make_layout(64))
    _scratch_74 = _worker_scratch.phase_74.body.get_tensor(cute.make_layout(8))
    _scratch_80 = _worker_scratch.phase_80.body.get_tensor(cute.make_layout(8))
    _scratch_84 = _worker_scratch.phase_84.body.get_tensor(cute.make_layout(8))
    _scratch_85 = _worker_scratch.phase_85.body.get_tensor(cute.make_layout(8))
    _scratch_88 = _worker_scratch.phase_88.body.get_tensor(cute.make_layout(64))
    _scratch_92 = _worker_scratch.phase_92.body.get_tensor(cute.make_layout(8))
    _scratch_98 = _worker_scratch.phase_98.body.get_tensor(cute.make_layout(8))
    _scratch_102 = _worker_scratch.phase_102.body.get_tensor(cute.make_layout(8))
    _scratch_103 = _worker_scratch.phase_103.body.get_tensor(cute.make_layout(8))
    _scratch_106 = _worker_scratch.phase_106.body.get_tensor(cute.make_layout(64))
    _scratch_110 = _worker_scratch.phase_110.body.get_tensor(cute.make_layout(8))
    _scratch_116 = _worker_scratch.phase_116.body.get_tensor(cute.make_layout(8))
    if _thread == 0:
        _worker_scratch.phase_10.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_21.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_23.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_38.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_44.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_48.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_49.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_56.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_62.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_66.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_67.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_74.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_80.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_84.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_85.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_92.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_98.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_102.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_103.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_110.barrier_0[0] = 0
    cute.arch.sync_threads()
    if _thread == 0:
        _worker_scratch.phase_116.barrier_0[0] = 0
    cute.arch.sync_threads()

    _g0_scratch_0 = cutlass.utils.SmemAllocator().allocate(_g0_body_13.shared_storage)
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
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 5120
            remainder_0 = remainder_0 // 5120
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 else cutlass.Int64(0))
            if load_0_0 < 0:
                llvm.intr_trap()
            if load_0_0 >= 151936:
                llvm.intr_trap()
            load_1_0 = (tensor_bindings[1][(tensor_bindings[0][0, o1_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 else cutlass.Int64(0)), o2_0] if 0 <= (o2_0) < 5120 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 5120
            remainder_1 = remainder_1 // 5120
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 else cutlass.Int64(0))
            if load_0_1 < 0:
                llvm.intr_trap()
            if load_0_1 >= 151936:
                llvm.intr_trap()
            load_1_1 = (tensor_bindings[1][(tensor_bindings[0][0, o1_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 else cutlass.Int64(0)), o2_1] if 0 <= (o2_1) < 5120 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 5120
            remainder_2 = remainder_2 // 5120
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 else cutlass.Int64(0))
            if load_0_2 < 0:
                llvm.intr_trap()
            if load_0_2 >= 151936:
                llvm.intr_trap()
            load_1_2 = (tensor_bindings[1][(tensor_bindings[0][0, o1_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 else cutlass.Int64(0)), o2_2] if 0 <= (o2_2) < 5120 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 5120
            remainder_3 = remainder_3 // 5120
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 else cutlass.Int64(0))
            if load_0_3 < 0:
                llvm.intr_trap()
            if load_0_3 >= 151936:
                llvm.intr_trap()
            load_1_3 = (tensor_bindings[1][(tensor_bindings[0][0, o1_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 else cutlass.Int64(0)), o2_3] if 0 <= (o2_3) < 5120 else cutlass.BFloat16(0))
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            row_index = work_iterator[tile]
            if row_index < 4:
                o1 = (row_index // 1) % 4
                o0 = (row_index // 4) % 1
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for reduction_index in cutlass.range(thread, 5120, 256):
                    r0 = reduction_index
                    accumulator = (accumulator + cute.pow(((tensor_bindings[0][o0, o1, r0] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (r0) < 5120 else cutlass.BFloat16(0))).to(cutlass.Float32), (cutlass.Int64(2)).to(cutlass.Float32)))
                warp_partial = cute.arch.warp_reduction_sum(accumulator)
                if lane == 0:
                    scratch[ warp] = warp_partial
                cute.arch.sync_threads()
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for warp_index in cutlass.range(warps):
                    accumulator = accumulator + scratch[ warp_index]
                for post_index in cutlass.range(thread, 5120, 256):
                    o2 = (post_index // 1) % 5120
                    post_load_0 = (tensor_bindings[0][o0, o1, o2] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 5120 else cutlass.BFloat16(0))
                    post_load_1 = (tensor_bindings[1][o2] if 0 <= (o2) < 5120 else cutlass.BFloat16(0))
                    output[o0, o1, o2] = (((post_load_1).to(cutlass.Float32) * ((((post_load_0).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(5120)).to(cutlass.Float32)) + cutlass.Float32(1e-06))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
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
            base = work_iterator[tile] + thread
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            load_1_0 = (tensor_bindings[1][0, 0, 0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_0) < 4 else cutlass.Int64(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            load_1_1 = (tensor_bindings[1][0, 0, 0, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_1) < 4 else cutlass.Int64(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            load_1_2 = (tensor_bindings[1][0, 0, 0, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_2) < 4 else cutlass.Int64(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            load_1_3 = (tensor_bindings[1][0, 0, 0, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o3_3) < 4 else cutlass.Int64(0))
            value_0 = (load_0_0 - load_1_0)
            value_1 = (load_0_1 - load_1_1)
            value_2 = (load_0_2 - load_1_2)
            value_3 = (load_0_3 - load_1_3)
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
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o3_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o3_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o3_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o3_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (0) < 1 else cutlass.Int64(0))
            value_0 = (load_0_0 >= cutlass.Int64(0))
            value_1 = (load_0_1 >= cutlass.Int64(0))
            value_2 = (load_0_2 >= cutlass.Int64(0))
            value_3 = (load_0_3 >= cutlass.Int64(0))
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
    def _body_12(
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
            base = work_iterator[tile] + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o2_0 = remainder_0 % 64
            remainder_0 = remainder_0 // 64
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, 0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_0 = (tensor_bindings[1][o2_0] if 0 <= (o2_0) < 64 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, 0] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_1 = (tensor_bindings[1][o2_1] if 0 <= (o2_1) < 64 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, 0] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_2 = (tensor_bindings[1][o2_2] if 0 <= (o2_2) < 64 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, 0] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (0) < 1 else cutlass.Float32(0))
            load_1_3 = (tensor_bindings[1][o2_3] if 0 <= (o2_3) < 64 else cutlass.Float32(0))
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
            o3_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 4 else cutlass.Int64(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 4 else cutlass.Int64(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 4 else cutlass.Int64(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 4 else cutlass.Int64(0))
            value_0 = cute.abs(load_0_0)
            value_1 = cute.abs(load_0_1)
            value_2 = cute.abs(load_0_2)
            value_3 = cute.abs(load_0_3)
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
            if row_index < 160:
                o2 = (row_index // 1) % 40
                o1 = (row_index // 40) % 4
                o0 = (row_index // 160) % 1
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for reduction_index in cutlass.range(thread, 128, 256, unroll_full=True):
                    r0 = reduction_index
                    accumulator = (accumulator + cute.pow(((tensor_bindings[0][o0, o1, o2, r0] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 40 and 0 <= (r0) < 128 else cutlass.BFloat16(0))).to(cutlass.Float32), (cutlass.Int64(2)).to(cutlass.Float32)))
                warp_partial = cute.arch.warp_reduction_sum(accumulator)
                if lane == 0:
                    scratch[ warp] = warp_partial
                cute.arch.sync_threads()
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for warp_index in cutlass.range(warps):
                    accumulator = accumulator + scratch[ warp_index]
                post_index_0 = thread + 0
                o3_0 = (post_index_0 // 1) % 128
                post_load_0_0 = (tensor_bindings[0][o0, o1, o2, o3_0] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 40 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
                post_load_1_0 = (tensor_bindings[1][o3_0] if 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
                if post_index_0 < 128:
                    output[o0, o1, o2, o3_0] = (((post_load_1_0).to(cutlass.Float32) * ((((post_load_0_0).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(128)).to(cutlass.Float32)) + cutlass.Float32(1e-06))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                cute.arch.sync_threads()

    @cute.jit
    def _body_17(
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
            o3_0 = remainder_0 % 64
            remainder_0 = remainder_0 // 64
            o2_0 = remainder_0 % 2
            remainder_0 = remainder_0 // 2
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 2 and 0 <= (o3_0) < 64 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 64
            remainder_1 = remainder_1 // 64
            o2_1 = remainder_1 % 2
            remainder_1 = remainder_1 // 2
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 2 and 0 <= (o3_1) < 64 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 64
            remainder_2 = remainder_2 // 64
            o2_2 = remainder_2 % 2
            remainder_2 = remainder_2 // 2
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 2 and 0 <= (o3_2) < 64 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 64
            remainder_3 = remainder_3 // 64
            o2_3 = remainder_3 % 2
            remainder_3 = remainder_3 // 2
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 2 and 0 <= (o3_3) < 64 else cutlass.Float32(0))
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
    def _body_18(
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
            if row_index < 32:
                o2 = (row_index // 1) % 8
                o1 = (row_index // 8) % 4
                o0 = (row_index // 32) % 1
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for reduction_index in cutlass.range(thread, 128, 256, unroll_full=True):
                    r0 = reduction_index
                    accumulator = (accumulator + cute.pow(((tensor_bindings[0][o0, o1, o2, r0] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 8 and 0 <= (r0) < 128 else cutlass.BFloat16(0))).to(cutlass.Float32), (cutlass.Int64(2)).to(cutlass.Float32)))
                warp_partial = cute.arch.warp_reduction_sum(accumulator)
                if lane == 0:
                    scratch[ warp] = warp_partial
                cute.arch.sync_threads()
                accumulator = cutlass.Float32(cutlass.Float32(0))
                for warp_index in cutlass.range(warps):
                    accumulator = accumulator + scratch[ warp_index]
                post_index_0 = thread + 0
                o3_0 = (post_index_0 // 1) % 128
                post_load_0_0 = (tensor_bindings[0][o0, o1, o2, o3_0] if 0 <= (o0) < 1 and 0 <= (o1) < 4 and 0 <= (o2) < 8 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
                post_load_1_0 = (tensor_bindings[1][o3_0] if 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
                if post_index_0 < 128:
                    output[o0, o1, o2, o3_0] = (((post_load_1_0).to(cutlass.Float32) * ((((post_load_0_0).to(cutlass.Float32) * cute.rsqrt(((accumulator / (cutlass.Int64(128)).to(cutlass.Float32)) + cutlass.Float32(1e-06))))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
                cute.arch.sync_threads()

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
            base = work_iterator[tile] + thread
            linear_0 = base + 0
            remainder_0 = linear_0
            o3_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            remainder_0 = remainder_0 // 1
            o1_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 4 else cutlass.Int64(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            remainder_1 = remainder_1 // 1
            o1_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 4 else cutlass.Int64(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            remainder_2 = remainder_2 // 1
            o1_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 4 else cutlass.Int64(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            remainder_3 = remainder_3 // 1
            o1_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 4 else cutlass.Int64(0))
            value_0 = (load_0_0 <= cutlass.Int64(4096))
            value_1 = (load_0_1 <= cutlass.Int64(4096))
            value_2 = (load_0_2 <= cutlass.Int64(4096))
            value_3 = (load_0_3 <= cutlass.Int64(4096))
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_20(
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
            o2_0 = remainder_0 % 128
            remainder_0 = remainder_0 // 128
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 128 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 128
            remainder_1 = remainder_1 // 128
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 128 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 128
            remainder_2 = remainder_2 // 128
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 128 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 128
            remainder_3 = remainder_3 // 128
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 128 else cutlass.Float32(0))
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
    def _body_21(
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
            o2_0 = remainder_0 % 128
            remainder_0 = remainder_0 // 128
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 128 else cutlass.Float32(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 128
            remainder_1 = remainder_1 // 128
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 128 else cutlass.Float32(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 128
            remainder_2 = remainder_2 // 128
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 128 else cutlass.Float32(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 128
            remainder_3 = remainder_3 // 128
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 128 else cutlass.Float32(0))
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
    def _body_22(
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
    def _body_23(
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
            o3_0 = remainder_0 % 128
            remainder_0 = remainder_0 // 128
            o1_0 = remainder_0 % 40
            remainder_0 = remainder_0 // 40
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0, (64 + o3_0)] if 0 <= (0) < 1 and 0 <= (o1_0) < 40 and 0 <= (o2_0) < 4 and 0 <= ((64 + o3_0)) < 128 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[0][0, o1_0, o2_0, (-64 + o3_0)] if 0 <= (0) < 1 and 0 <= (o1_0) < 40 and 0 <= (o2_0) < 4 and 0 <= ((-64 + o3_0)) < 128 else cutlass.BFloat16(0))
            load_2_0 = (tensor_bindings[1][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
            load_3_0 = (tensor_bindings[0][0, o1_0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 40 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
            load_4_0 = (tensor_bindings[2][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 128
            remainder_1 = remainder_1 // 128
            o1_1 = remainder_1 % 40
            remainder_1 = remainder_1 // 40
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1, (64 + o3_1)] if 0 <= (0) < 1 and 0 <= (o1_1) < 40 and 0 <= (o2_1) < 4 and 0 <= ((64 + o3_1)) < 128 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[0][0, o1_1, o2_1, (-64 + o3_1)] if 0 <= (0) < 1 and 0 <= (o1_1) < 40 and 0 <= (o2_1) < 4 and 0 <= ((-64 + o3_1)) < 128 else cutlass.BFloat16(0))
            load_2_1 = (tensor_bindings[1][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 128 else cutlass.BFloat16(0))
            load_3_1 = (tensor_bindings[0][0, o1_1, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 40 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 128 else cutlass.BFloat16(0))
            load_4_1 = (tensor_bindings[2][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 128 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 128
            remainder_2 = remainder_2 // 128
            o1_2 = remainder_2 % 40
            remainder_2 = remainder_2 // 40
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2, (64 + o3_2)] if 0 <= (0) < 1 and 0 <= (o1_2) < 40 and 0 <= (o2_2) < 4 and 0 <= ((64 + o3_2)) < 128 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[0][0, o1_2, o2_2, (-64 + o3_2)] if 0 <= (0) < 1 and 0 <= (o1_2) < 40 and 0 <= (o2_2) < 4 and 0 <= ((-64 + o3_2)) < 128 else cutlass.BFloat16(0))
            load_2_2 = (tensor_bindings[1][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 128 else cutlass.BFloat16(0))
            load_3_2 = (tensor_bindings[0][0, o1_2, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 40 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 128 else cutlass.BFloat16(0))
            load_4_2 = (tensor_bindings[2][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 128 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 128
            remainder_3 = remainder_3 // 128
            o1_3 = remainder_3 % 40
            remainder_3 = remainder_3 // 40
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3, (64 + o3_3)] if 0 <= (0) < 1 and 0 <= (o1_3) < 40 and 0 <= (o2_3) < 4 and 0 <= ((64 + o3_3)) < 128 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[0][0, o1_3, o2_3, (-64 + o3_3)] if 0 <= (0) < 1 and 0 <= (o1_3) < 40 and 0 <= (o2_3) < 4 and 0 <= ((-64 + o3_3)) < 128 else cutlass.BFloat16(0))
            load_2_3 = (tensor_bindings[1][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 128 else cutlass.BFloat16(0))
            load_3_3 = (tensor_bindings[0][0, o1_3, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 40 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 128 else cutlass.BFloat16(0))
            load_4_3 = (tensor_bindings[2][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 128 else cutlass.BFloat16(0))
            value_0 = ((((((load_3_0).to(cutlass.Float32) * (load_4_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_0))) if ((o3_0 < cutlass.Int64(64))) else (load_1_0))).to(cutlass.Float32) * (load_2_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_1 = ((((((load_3_1).to(cutlass.Float32) * (load_4_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_1))) if ((o3_1 < cutlass.Int64(64))) else (load_1_1))).to(cutlass.Float32) * (load_2_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_2 = ((((((load_3_2).to(cutlass.Float32) * (load_4_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_2))) if ((o3_2 < cutlass.Int64(64))) else (load_1_2))).to(cutlass.Float32) * (load_2_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_3 = ((((((load_3_3).to(cutlass.Float32) * (load_4_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_3))) if ((o3_3 < cutlass.Int64(64))) else (load_1_3))).to(cutlass.Float32) * (load_2_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

    @cute.jit
    def _body_24(
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
            o3_0 = remainder_0 % 128
            remainder_0 = remainder_0 // 128
            o1_0 = remainder_0 % 8
            remainder_0 = remainder_0 // 8
            o2_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0, (64 + o3_0)] if 0 <= (0) < 1 and 0 <= (o1_0) < 8 and 0 <= (o2_0) < 4 and 0 <= ((64 + o3_0)) < 128 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[0][0, o1_0, o2_0, (-64 + o3_0)] if 0 <= (0) < 1 and 0 <= (o1_0) < 8 and 0 <= (o2_0) < 4 and 0 <= ((-64 + o3_0)) < 128 else cutlass.BFloat16(0))
            load_2_0 = (tensor_bindings[1][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
            load_3_0 = (tensor_bindings[0][0, o1_0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 8 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
            load_4_0 = (tensor_bindings[2][0, 0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 128
            remainder_1 = remainder_1 // 128
            o1_1 = remainder_1 % 8
            remainder_1 = remainder_1 // 8
            o2_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1, (64 + o3_1)] if 0 <= (0) < 1 and 0 <= (o1_1) < 8 and 0 <= (o2_1) < 4 and 0 <= ((64 + o3_1)) < 128 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[0][0, o1_1, o2_1, (-64 + o3_1)] if 0 <= (0) < 1 and 0 <= (o1_1) < 8 and 0 <= (o2_1) < 4 and 0 <= ((-64 + o3_1)) < 128 else cutlass.BFloat16(0))
            load_2_1 = (tensor_bindings[1][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 128 else cutlass.BFloat16(0))
            load_3_1 = (tensor_bindings[0][0, o1_1, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 8 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 128 else cutlass.BFloat16(0))
            load_4_1 = (tensor_bindings[2][0, 0, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (o3_1) < 128 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 128
            remainder_2 = remainder_2 // 128
            o1_2 = remainder_2 % 8
            remainder_2 = remainder_2 // 8
            o2_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2, (64 + o3_2)] if 0 <= (0) < 1 and 0 <= (o1_2) < 8 and 0 <= (o2_2) < 4 and 0 <= ((64 + o3_2)) < 128 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[0][0, o1_2, o2_2, (-64 + o3_2)] if 0 <= (0) < 1 and 0 <= (o1_2) < 8 and 0 <= (o2_2) < 4 and 0 <= ((-64 + o3_2)) < 128 else cutlass.BFloat16(0))
            load_2_2 = (tensor_bindings[1][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 128 else cutlass.BFloat16(0))
            load_3_2 = (tensor_bindings[0][0, o1_2, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 8 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 128 else cutlass.BFloat16(0))
            load_4_2 = (tensor_bindings[2][0, 0, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (o3_2) < 128 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 128
            remainder_3 = remainder_3 // 128
            o1_3 = remainder_3 % 8
            remainder_3 = remainder_3 // 8
            o2_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3, (64 + o3_3)] if 0 <= (0) < 1 and 0 <= (o1_3) < 8 and 0 <= (o2_3) < 4 and 0 <= ((64 + o3_3)) < 128 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[0][0, o1_3, o2_3, (-64 + o3_3)] if 0 <= (0) < 1 and 0 <= (o1_3) < 8 and 0 <= (o2_3) < 4 and 0 <= ((-64 + o3_3)) < 128 else cutlass.BFloat16(0))
            load_2_3 = (tensor_bindings[1][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 128 else cutlass.BFloat16(0))
            load_3_3 = (tensor_bindings[0][0, o1_3, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 8 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 128 else cutlass.BFloat16(0))
            load_4_3 = (tensor_bindings[2][0, 0, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (o3_3) < 128 else cutlass.BFloat16(0))
            value_0 = ((((((load_3_0).to(cutlass.Float32) * (load_4_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_0))) if ((o3_0 < cutlass.Int64(64))) else (load_1_0))).to(cutlass.Float32) * (load_2_0).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_1 = ((((((load_3_1).to(cutlass.Float32) * (load_4_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_1))) if ((o3_1 < cutlass.Int64(64))) else (load_1_1))).to(cutlass.Float32) * (load_2_1).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_2 = ((((((load_3_2).to(cutlass.Float32) * (load_4_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_2))) if ((o3_2 < cutlass.Int64(64))) else (load_1_2))).to(cutlass.Float32) * (load_2_2).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            value_3 = ((((((load_3_3).to(cutlass.Float32) * (load_4_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32) + (((((((-(load_0_3))) if ((o3_3 < cutlass.Int64(64))) else (load_1_3))).to(cutlass.Float32) * (load_2_3).to(cutlass.Float32))).to(cutlass.BFloat16)).to(cutlass.Float32))).to(cutlass.BFloat16)
            if linear_0 < output_count:
                output[o0_0, o1_0, o2_0, o3_0] = value_0
            if linear_1 < output_count:
                output[o0_1, o1_1, o2_1, o3_1] = value_1
            if linear_2 < output_count:
                output[o0_2, o1_2, o2_2, o3_2] = value_2
            if linear_3 < output_count:
                output[o0_3, o1_3, o2_3, o3_3] = value_3

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
            base = work_iterator[tile] + thread
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
            load_1_0 = (tensor_bindings[1][0, 0, o2_0, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_0) < 4 and 0 <= (0) < 1 else cutlass.Boolean(0))
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
            load_1_1 = (tensor_bindings[1][0, 0, o2_1, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_1) < 4 and 0 <= (0) < 1 else cutlass.Boolean(0))
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
            load_1_2 = (tensor_bindings[1][0, 0, o2_2, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_2) < 4 and 0 <= (0) < 1 else cutlass.Boolean(0))
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
            load_1_3 = (tensor_bindings[1][0, 0, o2_3, 0] if 0 <= (0) < 1 and 0 <= (0) < 1 and 0 <= (o2_3) < 4 and 0 <= (0) < 1 else cutlass.Boolean(0))
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
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            base = work_iterator[tile] + thread
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
    def _body_27(
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
    def _body_28(
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
        row_count = 160
        for tile in cutlass.range(block, cute.size(work_iterator), workers):
            row = work_iterator[tile] * 8 + warp
            if warp < 8:
                if row < row_count:
                    bh = row // 4
                    s = row % 4
                    b = bh // 40
                    h = bh % 40
                    kv_head = h // 5
                    base = warp * 4
                    maximum = cutlass.Float32(-1.0e30)
                    for key in cutlass.range(lane, 4, 32):
                        score = cutlass.Float32(0.0)
                        for d in cutlass.range_constexpr(128):
                            score = score + tensor_bindings[0][b, h, s, d].to(cutlass.Float32) * tensor_bindings[1][b, kv_head, key, d].to(cutlass.Float32)
                        score = score.to(cutlass.BFloat16).to(cutlass.Float32)
                        score = (score * cutlass.Float32(0.08838834764831845)).to(cutlass.BFloat16)
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
                    for d in cutlass.range(lane, 128, 32):
                        value = cutlass.Float32(0.0)
                        for key in cutlass.range(4, unroll=8):
                            value = value + scratch[base + key] * tensor_bindings[2][b, key, kv_head, d].to(cutlass.Float32)
                        tensor_bindings[4][bh, s, d] = value.to(cutlass.BFloat16)
                    cute.arch.sync_warp()

    @cute.jit
    def _body_29(
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
            o3_0 = remainder_0 % 128
            remainder_0 = remainder_0 // 128
            o2_0 = remainder_0 % 40
            remainder_0 = remainder_0 // 40
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0, o3_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 40 and 0 <= (o3_0) < 128 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o3_1 = remainder_1 % 128
            remainder_1 = remainder_1 // 128
            o2_1 = remainder_1 % 40
            remainder_1 = remainder_1 // 40
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1, o3_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 40 and 0 <= (o3_1) < 128 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o3_2 = remainder_2 % 128
            remainder_2 = remainder_2 // 128
            o2_2 = remainder_2 % 40
            remainder_2 = remainder_2 // 40
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2, o3_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 40 and 0 <= (o3_2) < 128 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o3_3 = remainder_3 % 128
            remainder_3 = remainder_3 // 128
            o2_3 = remainder_3 % 40
            remainder_3 = remainder_3 // 40
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3, o3_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 40 and 0 <= (o3_3) < 128 else cutlass.BFloat16(0))
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
    def _body_30(
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
            o2_0 = remainder_0 % 5120
            remainder_0 = remainder_0 // 5120
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 5120 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[1][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 5120 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 5120
            remainder_1 = remainder_1 // 5120
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 5120 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[1][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 5120 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 5120
            remainder_2 = remainder_2 // 5120
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 5120 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[1][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 5120 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 5120
            remainder_3 = remainder_3 // 5120
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 5120 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[1][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 5120 else cutlass.BFloat16(0))
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
    def _body_31(
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
            o2_0 = remainder_0 % 17408
            remainder_0 = remainder_0 // 17408
            o1_0 = remainder_0 % 4
            remainder_0 = remainder_0 // 4
            o0_0 = remainder_0 % 1
            load_0_0 = (tensor_bindings[0][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 17408 else cutlass.BFloat16(0))
            load_1_0 = (tensor_bindings[1][0, o1_0, o2_0] if 0 <= (0) < 1 and 0 <= (o1_0) < 4 and 0 <= (o2_0) < 17408 else cutlass.BFloat16(0))
            linear_1 = base + 256
            remainder_1 = linear_1
            o2_1 = remainder_1 % 17408
            remainder_1 = remainder_1 // 17408
            o1_1 = remainder_1 % 4
            remainder_1 = remainder_1 // 4
            o0_1 = remainder_1 % 1
            load_0_1 = (tensor_bindings[0][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 17408 else cutlass.BFloat16(0))
            load_1_1 = (tensor_bindings[1][0, o1_1, o2_1] if 0 <= (0) < 1 and 0 <= (o1_1) < 4 and 0 <= (o2_1) < 17408 else cutlass.BFloat16(0))
            linear_2 = base + 512
            remainder_2 = linear_2
            o2_2 = remainder_2 % 17408
            remainder_2 = remainder_2 // 17408
            o1_2 = remainder_2 % 4
            remainder_2 = remainder_2 // 4
            o0_2 = remainder_2 % 1
            load_0_2 = (tensor_bindings[0][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 17408 else cutlass.BFloat16(0))
            load_1_2 = (tensor_bindings[1][0, o1_2, o2_2] if 0 <= (0) < 1 and 0 <= (o1_2) < 4 and 0 <= (o2_2) < 17408 else cutlass.BFloat16(0))
            linear_3 = base + 768
            remainder_3 = linear_3
            o2_3 = remainder_3 % 17408
            remainder_3 = remainder_3 // 17408
            o1_3 = remainder_3 % 4
            remainder_3 = remainder_3 // 4
            o0_3 = remainder_3 % 1
            load_0_3 = (tensor_bindings[0][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 17408 else cutlass.BFloat16(0))
            load_1_3 = (tensor_bindings[1][0, o1_3, o2_3] if 0 <= (0) < 1 and 0 <= (o1_3) < 4 and 0 <= (o2_3) < 17408 else cutlass.BFloat16(0))
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



    @cute.jit
    def _no_op():
        pass

    @cute.jit
    def _ignore_epoch(epoch: cutlass.Int32):
        pass

    @cute.jit
    def _cta_sync(epoch: cutlass.Int32):
        cute.arch.sync_threads()

    @cute.jit
    def _run_terminal(phase_index: cutlass.Int32):
        pass

    @cute.jit
    def _run_variant_0(phase_index: cutlass.Int32):
        if phase_index < 1:
            if phase_index == 0:
                _body_0(
            work_iterators[0],
            _block,
            _thread,
            workers,
            (tensors[0],),
            None,
            None,
        )
        else:
            if phase_index < 2:
                if phase_index == 1:
                    _body_0(
            work_iterators[1],
            _block,
            _thread,
            workers,
            (tensors[1],),
            None,
            None,
        )
            else:
                if phase_index == 2:
                    _body_0(
            work_iterators[2],
            _block,
            _thread,
            workers,
            (tensors[2],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_1(phase_index: cutlass.Int32):
        _body_1(
            work_iterators[3],
            _block,
            _thread,
            workers,
            (tensors[3],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_2(phase_index: cutlass.Int32):
        _body_2(
            work_iterators[4],
            _block,
            _thread,
            workers,
            (tensors[118], tensors[119], tensors[4],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_3(phase_index: cutlass.Int32):
        if phase_index < 6:
            if phase_index == 5:
                _body_3(
            work_iterators[5],
            _block,
            _thread,
            workers,
            (tensors[5],),
            None,
            None,
        )
        else:
            if phase_index == 6:
                _body_3(
            work_iterators[6],
            _block,
            _thread,
            workers,
            (tensors[6],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_4(phase_index: cutlass.Int32):
        _body_4(
            work_iterators[7],
            _block,
            _thread,
            workers,
            (tensors[120], tensors[7],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_5(phase_index: cutlass.Int32):
        _body_5(
            work_iterators[8],
            _block,
            _thread,
            workers,
            (tensors[8],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_6(phase_index: cutlass.Int32):
        _body_6(
            work_iterators[9],
            _block,
            _thread,
            workers,
            (tensors[9],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_7(phase_index: cutlass.Int32):
        if phase_index < 74:
            if phase_index < 44:
                if phase_index < 38:
                    if phase_index == 10:
                        _body_7(
            work_iterators[10],
            _block,
            _thread,
            workers,
            (tensors[4], tensors[122], tensors[10],),
            _scratch_10,
            None,
        )
                else:
                    if phase_index == 38:
                        _body_7(
            work_iterators[34],
            _block,
            _thread,
            workers,
            (tensors[37], tensors[129], tensors[38],),
            _scratch_38,
            None,
        )
            else:
                if phase_index < 56:
                    if phase_index == 44:
                        _body_7(
            work_iterators[37],
            _block,
            _thread,
            workers,
            (tensors[43], tensors[133], tensors[44],),
            _scratch_44,
            None,
        )
                else:
                    if phase_index < 62:
                        if phase_index == 56:
                            _body_7(
            work_iterators[45],
            _block,
            _thread,
            workers,
            (tensors[55], tensors[140], tensors[56],),
            _scratch_56,
            None,
        )
                    else:
                        if phase_index == 62:
                            _body_7(
            work_iterators[48],
            _block,
            _thread,
            workers,
            (tensors[61], tensors[144], tensors[62],),
            _scratch_62,
            None,
        )
        else:
            if phase_index < 98:
                if phase_index < 80:
                    if phase_index == 74:
                        _body_7(
            work_iterators[56],
            _block,
            _thread,
            workers,
            (tensors[73], tensors[151], tensors[74],),
            _scratch_74,
            None,
        )
                else:
                    if phase_index < 92:
                        if phase_index == 80:
                            _body_7(
            work_iterators[59],
            _block,
            _thread,
            workers,
            (tensors[79], tensors[155], tensors[80],),
            _scratch_80,
            None,
        )
                    else:
                        if phase_index == 92:
                            _body_7(
            work_iterators[67],
            _block,
            _thread,
            workers,
            (tensors[91], tensors[162], tensors[92],),
            _scratch_92,
            None,
        )
            else:
                if phase_index < 110:
                    if phase_index == 98:
                        _body_7(
            work_iterators[70],
            _block,
            _thread,
            workers,
            (tensors[97], tensors[166], tensors[98],),
            _scratch_98,
            None,
        )
                else:
                    if phase_index < 116:
                        if phase_index == 110:
                            _body_7(
            work_iterators[78],
            _block,
            _thread,
            workers,
            (tensors[109], tensors[173], tensors[110],),
            _scratch_110,
            None,
        )
                    else:
                        if phase_index == 116:
                            _body_7(
            work_iterators[81],
            _block,
            _thread,
            workers,
            (tensors[115], tensors[177], tensors[116],),
            _scratch_116,
            None,
        )
    @cute.jit
    def _run_variant_8(phase_index: cutlass.Int32):
        _body_8(
            work_iterators[11],
            _block,
            _thread,
            workers,
            (tensors[238], tensors[11],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_9(phase_index: cutlass.Int32):
        _body_9(
            work_iterators[12],
            _block,
            _thread,
            workers,
            (tensors[239], tensors[240], tensors[12],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_10(phase_index: cutlass.Int32):
        _body_10(
            work_iterators[13],
            _block,
            _thread,
            workers,
            (tensors[239], tensors[13],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_11(phase_index: cutlass.Int32):
        _body_11(
            work_iterators[14],
            _block,
            _thread,
            workers,
            (tensors[241], tensors[240], tensors[7], tensors[14],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_12(phase_index: cutlass.Int32):
        _body_12(
            work_iterators[15],
            _block,
            _thread,
            workers,
            (tensors[3], tensors[15],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_13(phase_index: cutlass.Int32):
        if phase_index < 75:
            if phase_index < 47:
                if phase_index < 39:
                    if phase_index < 20:
                        if phase_index < 18:
                            if phase_index == 16:
                                _g0_body_13.run(
            g0_values[0][2],
            g0_values[0][0],
            _g0_scratch_0,
            g0_values[0][1],
        )
                        else:
                            if phase_index == 18:
                                _g0_body_13.run(
            g0_values[1][2],
            g0_values[1][0],
            _g0_scratch_0,
            g0_values[1][1],
        )
                    else:
                        if phase_index < 36:
                            if phase_index == 20:
                                _g0_body_13.run(
            g0_values[2][2],
            g0_values[2][0],
            _g0_scratch_0,
            g0_values[2][1],
        )
                        else:
                            if phase_index == 36:
                                _g0_body_13.run(
            g0_values[3][2],
            g0_values[3][0],
            _g0_scratch_0,
            g0_values[3][1],
        )
                else:
                    if phase_index < 42:
                        if phase_index < 40:
                            if phase_index == 39:
                                _g0_body_13.run(
            g0_values[4][2],
            g0_values[4][0],
            _g0_scratch_0,
            g0_values[4][1],
        )
                        else:
                            if phase_index == 40:
                                _g0_body_13.run(
            g0_values[5][2],
            g0_values[5][0],
            _g0_scratch_0,
            g0_values[5][1],
        )
                    else:
                        if phase_index < 45:
                            if phase_index == 42:
                                _g0_body_13.run(
            g0_values[6][2],
            g0_values[6][0],
            _g0_scratch_0,
            g0_values[6][1],
        )
                        else:
                            if phase_index < 46:
                                if phase_index == 45:
                                    _g0_body_13.run(
            g0_values[7][2],
            g0_values[7][0],
            _g0_scratch_0,
            g0_values[7][1],
        )
                            else:
                                if phase_index == 46:
                                    _g0_body_13.run(
            g0_values[8][2],
            g0_values[8][0],
            _g0_scratch_0,
            g0_values[8][1],
        )
            else:
                if phase_index < 60:
                    if phase_index < 57:
                        if phase_index < 54:
                            if phase_index == 47:
                                _g0_body_13.run(
            g0_values[9][2],
            g0_values[9][0],
            _g0_scratch_0,
            g0_values[9][1],
        )
                        else:
                            if phase_index == 54:
                                _g0_body_13.run(
            g0_values[10][2],
            g0_values[10][0],
            _g0_scratch_0,
            g0_values[10][1],
        )
                    else:
                        if phase_index < 58:
                            if phase_index == 57:
                                _g0_body_13.run(
            g0_values[11][2],
            g0_values[11][0],
            _g0_scratch_0,
            g0_values[11][1],
        )
                        else:
                            if phase_index == 58:
                                _g0_body_13.run(
            g0_values[12][2],
            g0_values[12][0],
            _g0_scratch_0,
            g0_values[12][1],
        )
                else:
                    if phase_index < 64:
                        if phase_index < 63:
                            if phase_index == 60:
                                _g0_body_13.run(
            g0_values[13][2],
            g0_values[13][0],
            _g0_scratch_0,
            g0_values[13][1],
        )
                        else:
                            if phase_index == 63:
                                _g0_body_13.run(
            g0_values[14][2],
            g0_values[14][0],
            _g0_scratch_0,
            g0_values[14][1],
        )
                    else:
                        if phase_index < 65:
                            if phase_index == 64:
                                _g0_body_13.run(
            g0_values[15][2],
            g0_values[15][0],
            _g0_scratch_0,
            g0_values[15][1],
        )
                        else:
                            if phase_index < 72:
                                if phase_index == 65:
                                    _g0_body_13.run(
            g0_values[16][2],
            g0_values[16][0],
            _g0_scratch_0,
            g0_values[16][1],
        )
                            else:
                                if phase_index == 72:
                                    _g0_body_13.run(
            g0_values[17][2],
            g0_values[17][0],
            _g0_scratch_0,
            g0_values[17][1],
        )
        else:
            if phase_index < 96:
                if phase_index < 82:
                    if phase_index < 78:
                        if phase_index < 76:
                            if phase_index == 75:
                                _g0_body_13.run(
            g0_values[18][2],
            g0_values[18][0],
            _g0_scratch_0,
            g0_values[18][1],
        )
                        else:
                            if phase_index == 76:
                                _g0_body_13.run(
            g0_values[19][2],
            g0_values[19][0],
            _g0_scratch_0,
            g0_values[19][1],
        )
                    else:
                        if phase_index < 81:
                            if phase_index == 78:
                                _g0_body_13.run(
            g0_values[20][2],
            g0_values[20][0],
            _g0_scratch_0,
            g0_values[20][1],
        )
                        else:
                            if phase_index == 81:
                                _g0_body_13.run(
            g0_values[21][2],
            g0_values[21][0],
            _g0_scratch_0,
            g0_values[21][1],
        )
                else:
                    if phase_index < 90:
                        if phase_index < 83:
                            if phase_index == 82:
                                _g0_body_13.run(
            g0_values[22][2],
            g0_values[22][0],
            _g0_scratch_0,
            g0_values[22][1],
        )
                        else:
                            if phase_index == 83:
                                _g0_body_13.run(
            g0_values[23][2],
            g0_values[23][0],
            _g0_scratch_0,
            g0_values[23][1],
        )
                    else:
                        if phase_index < 93:
                            if phase_index == 90:
                                _g0_body_13.run(
            g0_values[24][2],
            g0_values[24][0],
            _g0_scratch_0,
            g0_values[24][1],
        )
                        else:
                            if phase_index < 94:
                                if phase_index == 93:
                                    _g0_body_13.run(
            g0_values[25][2],
            g0_values[25][0],
            _g0_scratch_0,
            g0_values[25][1],
        )
                            else:
                                if phase_index == 94:
                                    _g0_body_13.run(
            g0_values[26][2],
            g0_values[26][0],
            _g0_scratch_0,
            g0_values[26][1],
        )
            else:
                if phase_index < 108:
                    if phase_index < 100:
                        if phase_index < 99:
                            if phase_index == 96:
                                _g0_body_13.run(
            g0_values[27][2],
            g0_values[27][0],
            _g0_scratch_0,
            g0_values[27][1],
        )
                        else:
                            if phase_index == 99:
                                _g0_body_13.run(
            g0_values[28][2],
            g0_values[28][0],
            _g0_scratch_0,
            g0_values[28][1],
        )
                    else:
                        if phase_index < 101:
                            if phase_index == 100:
                                _g0_body_13.run(
            g0_values[29][2],
            g0_values[29][0],
            _g0_scratch_0,
            g0_values[29][1],
        )
                        else:
                            if phase_index == 101:
                                _g0_body_13.run(
            g0_values[30][2],
            g0_values[30][0],
            _g0_scratch_0,
            g0_values[30][1],
        )
                else:
                    if phase_index < 112:
                        if phase_index < 111:
                            if phase_index == 108:
                                _g0_body_13.run(
            g0_values[31][2],
            g0_values[31][0],
            _g0_scratch_0,
            g0_values[31][1],
        )
                        else:
                            if phase_index == 111:
                                _g0_body_13.run(
            g0_values[32][2],
            g0_values[32][0],
            _g0_scratch_0,
            g0_values[32][1],
        )
                    else:
                        if phase_index < 114:
                            if phase_index == 112:
                                _g0_body_13.run(
            g0_values[33][2],
            g0_values[33][0],
            _g0_scratch_0,
            g0_values[33][1],
        )
                        else:
                            if phase_index < 117:
                                if phase_index == 114:
                                    _g0_body_13.run(
            g0_values[34][2],
            g0_values[34][0],
            _g0_scratch_0,
            g0_values[34][1],
        )
                            else:
                                if phase_index == 117:
                                    _g0_body_13.run(
            g0_values[35][2],
            g0_values[35][0],
            _g0_scratch_0,
            g0_values[35][1],
        )
    @cute.jit
    def _run_variant_14(phase_index: cutlass.Int32):
        _body_14(
            work_iterators[16],
            _block,
            _thread,
            workers,
            (tensors[11], tensors[121], tensors[17],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_15(phase_index: cutlass.Int32):
        _body_15(
            work_iterators[17],
            _block,
            _thread,
            workers,
            (tensors[12], tensors[19],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_16(phase_index: cutlass.Int32):
        if phase_index < 66:
            if phase_index < 48:
                if phase_index == 21:
                    _body_16(
            work_iterators[18],
            _block,
            _thread,
            workers,
            (tensors[248], tensors[124], tensors[21],),
            _scratch_21,
            None,
        )
            else:
                if phase_index == 48:
                    _body_16(
            work_iterators[38],
            _block,
            _thread,
            workers,
            (tensors[277], tensors[135], tensors[48],),
            _scratch_48,
            None,
        )
        else:
            if phase_index < 84:
                if phase_index == 66:
                    _body_16(
            work_iterators[49],
            _block,
            _thread,
            workers,
            (tensors[303], tensors[146], tensors[66],),
            _scratch_66,
            None,
        )
            else:
                if phase_index < 102:
                    if phase_index == 84:
                        _body_16(
            work_iterators[60],
            _block,
            _thread,
            workers,
            (tensors[329], tensors[157], tensors[84],),
            _scratch_84,
            None,
        )
                else:
                    if phase_index == 102:
                        _body_16(
            work_iterators[71],
            _block,
            _thread,
            workers,
            (tensors[355], tensors[168], tensors[102],),
            _scratch_102,
            None,
        )
    @cute.jit
    def _run_variant_17(phase_index: cutlass.Int32):
        _body_17(
            work_iterators[19],
            _block,
            _thread,
            workers,
            (tensors[249], tensors[22],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_18(phase_index: cutlass.Int32):
        if phase_index < 67:
            if phase_index < 49:
                if phase_index == 23:
                    _body_18(
            work_iterators[20],
            _block,
            _thread,
            workers,
            (tensors[250], tensors[126], tensors[23],),
            _scratch_23,
            None,
        )
            else:
                if phase_index == 49:
                    _body_18(
            work_iterators[39],
            _block,
            _thread,
            workers,
            (tensors[278], tensors[137], tensors[49],),
            _scratch_49,
            None,
        )
        else:
            if phase_index < 85:
                if phase_index == 67:
                    _body_18(
            work_iterators[50],
            _block,
            _thread,
            workers,
            (tensors[304], tensors[148], tensors[67],),
            _scratch_67,
            None,
        )
            else:
                if phase_index < 103:
                    if phase_index == 85:
                        _body_18(
            work_iterators[61],
            _block,
            _thread,
            workers,
            (tensors[330], tensors[159], tensors[85],),
            _scratch_85,
            None,
        )
                else:
                    if phase_index == 103:
                        _body_18(
            work_iterators[72],
            _block,
            _thread,
            workers,
            (tensors[356], tensors[170], tensors[103],),
            _scratch_103,
            None,
        )
    @cute.jit
    def _run_variant_19(phase_index: cutlass.Int32):
        _body_19(
            work_iterators[21],
            _block,
            _thread,
            workers,
            (tensors[19], tensors[24],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_20(phase_index: cutlass.Int32):
        _body_20(
            work_iterators[22],
            _block,
            _thread,
            workers,
            (tensors[251], tensors[25],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_21(phase_index: cutlass.Int32):
        _body_21(
            work_iterators[23],
            _block,
            _thread,
            workers,
            (tensors[251], tensors[26],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_22(phase_index: cutlass.Int32):
        if phase_index < 31:
            if phase_index == 27:
                _body_22(
            work_iterators[24],
            _block,
            _thread,
            workers,
            (tensors[6], tensors[24], tensors[27],),
            None,
            None,
        )
        else:
            if phase_index == 31:
                _body_22(
            work_iterators[28],
            _block,
            _thread,
            workers,
            (tensors[5], tensors[30], tensors[31],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_23(phase_index: cutlass.Int32):
        if phase_index < 68:
            if phase_index < 50:
                if phase_index == 28:
                    _body_23(
            work_iterators[25],
            _block,
            _thread,
            workers,
            (tensors[252], tensors[253], tensors[254], tensors[28],),
            None,
            None,
        )
            else:
                if phase_index == 50:
                    _body_23(
            work_iterators[40],
            _block,
            _thread,
            workers,
            (tensors[279], tensors[280], tensors[281], tensors[50],),
            None,
            None,
        )
        else:
            if phase_index < 86:
                if phase_index == 68:
                    _body_23(
            work_iterators[51],
            _block,
            _thread,
            workers,
            (tensors[305], tensors[306], tensors[307], tensors[68],),
            None,
            None,
        )
            else:
                if phase_index < 104:
                    if phase_index == 86:
                        _body_23(
            work_iterators[62],
            _block,
            _thread,
            workers,
            (tensors[331], tensors[332], tensors[333], tensors[86],),
            None,
            None,
        )
                else:
                    if phase_index == 104:
                        _body_23(
            work_iterators[73],
            _block,
            _thread,
            workers,
            (tensors[357], tensors[358], tensors[359], tensors[104],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_24(phase_index: cutlass.Int32):
        if phase_index < 69:
            if phase_index < 51:
                if phase_index == 29:
                    _body_24(
            work_iterators[26],
            _block,
            _thread,
            workers,
            (tensors[255], tensors[253], tensors[254], tensors[29],),
            None,
            None,
        )
            else:
                if phase_index == 51:
                    _body_24(
            work_iterators[41],
            _block,
            _thread,
            workers,
            (tensors[282], tensors[280], tensors[281], tensors[51],),
            None,
            None,
        )
        else:
            if phase_index < 87:
                if phase_index == 69:
                    _body_24(
            work_iterators[52],
            _block,
            _thread,
            workers,
            (tensors[308], tensors[306], tensors[307], tensors[69],),
            None,
            None,
        )
            else:
                if phase_index < 105:
                    if phase_index == 87:
                        _body_24(
            work_iterators[63],
            _block,
            _thread,
            workers,
            (tensors[334], tensors[332], tensors[333], tensors[87],),
            None,
            None,
        )
                else:
                    if phase_index == 105:
                        _body_24(
            work_iterators[74],
            _block,
            _thread,
            workers,
            (tensors[360], tensors[358], tensors[359], tensors[105],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_25(phase_index: cutlass.Int32):
        _body_25(
            work_iterators[27],
            _block,
            _thread,
            workers,
            (tensors[27], tensors[13], tensors[30],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_26(phase_index: cutlass.Int32):
        _body_26(
            work_iterators[29],
            _block,
            _thread,
            workers,
            (tensors[31], tensors[14], tensors[32],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_27(phase_index: cutlass.Int32):
        _body_27(
            work_iterators[30],
            _block,
            _thread,
            workers,
            (tensors[256], tensors[15], tensors[9], tensors[33],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_28(phase_index: cutlass.Int32):
        if phase_index < 70:
            if phase_index < 52:
                if phase_index == 34:
                    _body_28(
            work_iterators[31],
            _block,
            _thread,
            workers,
            (tensors[28], tensors[29], tensors[257], tensors[32], tensors[34],),
            _scratch_34,
            None,
        )
            else:
                if phase_index == 52:
                    _body_28(
            work_iterators[42],
            _block,
            _thread,
            workers,
            (tensors[50], tensors[51], tensors[283], tensors[32], tensors[52],),
            _scratch_52,
            None,
        )
        else:
            if phase_index < 88:
                if phase_index == 70:
                    _body_28(
            work_iterators[53],
            _block,
            _thread,
            workers,
            (tensors[68], tensors[69], tensors[309], tensors[32], tensors[70],),
            _scratch_70,
            None,
        )
            else:
                if phase_index < 106:
                    if phase_index == 88:
                        _body_28(
            work_iterators[64],
            _block,
            _thread,
            workers,
            (tensors[86], tensors[87], tensors[335], tensors[32], tensors[88],),
            _scratch_88,
            None,
        )
                else:
                    if phase_index == 106:
                        _body_28(
            work_iterators[75],
            _block,
            _thread,
            workers,
            (tensors[104], tensors[105], tensors[361], tensors[32], tensors[106],),
            _scratch_106,
            None,
        )
    @cute.jit
    def _run_variant_29(phase_index: cutlass.Int32):
        if phase_index < 71:
            if phase_index < 53:
                if phase_index == 35:
                    _body_29(
            work_iterators[32],
            _block,
            _thread,
            workers,
            (tensors[258], tensors[35],),
            None,
            None,
        )
            else:
                if phase_index == 53:
                    _body_29(
            work_iterators[43],
            _block,
            _thread,
            workers,
            (tensors[284], tensors[53],),
            None,
            None,
        )
        else:
            if phase_index < 89:
                if phase_index == 71:
                    _body_29(
            work_iterators[54],
            _block,
            _thread,
            workers,
            (tensors[310], tensors[71],),
            None,
            None,
        )
            else:
                if phase_index < 107:
                    if phase_index == 89:
                        _body_29(
            work_iterators[65],
            _block,
            _thread,
            workers,
            (tensors[336], tensors[89],),
            None,
            None,
        )
                else:
                    if phase_index == 107:
                        _body_29(
            work_iterators[76],
            _block,
            _thread,
            workers,
            (tensors[362], tensors[107],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_30(phase_index: cutlass.Int32):
        if phase_index < 79:
            if phase_index < 55:
                if phase_index < 43:
                    if phase_index == 37:
                        _body_30(
            work_iterators[33],
            _block,
            _thread,
            workers,
            (tensors[4], tensors[261], tensors[37],),
            None,
            None,
        )
                else:
                    if phase_index == 43:
                        _body_30(
            work_iterators[36],
            _block,
            _thread,
            workers,
            (tensors[37], tensors[270], tensors[43],),
            None,
            None,
        )
            else:
                if phase_index < 61:
                    if phase_index == 55:
                        _body_30(
            work_iterators[44],
            _block,
            _thread,
            workers,
            (tensors[43], tensors[287], tensors[55],),
            None,
            None,
        )
                else:
                    if phase_index < 73:
                        if phase_index == 61:
                            _body_30(
            work_iterators[47],
            _block,
            _thread,
            workers,
            (tensors[55], tensors[296], tensors[61],),
            None,
            None,
        )
                    else:
                        if phase_index == 73:
                            _body_30(
            work_iterators[55],
            _block,
            _thread,
            workers,
            (tensors[61], tensors[313], tensors[73],),
            None,
            None,
        )
        else:
            if phase_index < 97:
                if phase_index < 91:
                    if phase_index == 79:
                        _body_30(
            work_iterators[58],
            _block,
            _thread,
            workers,
            (tensors[73], tensors[322], tensors[79],),
            None,
            None,
        )
                else:
                    if phase_index == 91:
                        _body_30(
            work_iterators[66],
            _block,
            _thread,
            workers,
            (tensors[79], tensors[339], tensors[91],),
            None,
            None,
        )
            else:
                if phase_index < 109:
                    if phase_index == 97:
                        _body_30(
            work_iterators[69],
            _block,
            _thread,
            workers,
            (tensors[91], tensors[348], tensors[97],),
            None,
            None,
        )
                else:
                    if phase_index < 115:
                        if phase_index == 109:
                            _body_30(
            work_iterators[77],
            _block,
            _thread,
            workers,
            (tensors[97], tensors[365], tensors[109],),
            None,
            None,
        )
                    else:
                        if phase_index == 115:
                            _body_30(
            work_iterators[80],
            _block,
            _thread,
            workers,
            (tensors[109], tensors[374], tensors[115],),
            None,
            None,
        )
    @cute.jit
    def _run_variant_31(phase_index: cutlass.Int32):
        if phase_index < 77:
            if phase_index < 59:
                if phase_index == 41:
                    _body_31(
            work_iterators[35],
            _block,
            _thread,
            workers,
            (tensors[266], tensors[267], tensors[41],),
            None,
            None,
        )
            else:
                if phase_index == 59:
                    _body_31(
            work_iterators[46],
            _block,
            _thread,
            workers,
            (tensors[292], tensors[293], tensors[59],),
            None,
            None,
        )
        else:
            if phase_index < 95:
                if phase_index == 77:
                    _body_31(
            work_iterators[57],
            _block,
            _thread,
            workers,
            (tensors[318], tensors[319], tensors[77],),
            None,
            None,
        )
            else:
                if phase_index < 113:
                    if phase_index == 95:
                        _body_31(
            work_iterators[68],
            _block,
            _thread,
            workers,
            (tensors[344], tensors[345], tensors[95],),
            None,
            None,
        )
                else:
                    if phase_index == 113:
                        _body_31(
            work_iterators[79],
            _block,
            _thread,
            workers,
            (tensors[370], tensors[371], tensors[113],),
            None,
            None,
        )

    @cute.jit
    def _publish_phase_9(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 9)
    @cute.jit
    def _publish_phase_15(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 15)
    @cute.jit
    def _publish_phase_20(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 20)
    @cute.jit
    def _publish_phase_24(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 24)
    @cute.jit
    def _publish_phase_27(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 27)
    @cute.jit
    def _publish_phase_30(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 30)
    @cute.jit
    def _publish_phase_31(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 31)
    @cute.jit
    def _publish_phase_32(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 32)
    @cute.jit
    def _publish_phase_34(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 34)
    @cute.jit
    def _publish_phase_35(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 35)
    @cute.jit
    def _publish_phase_36(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 36)
    @cute.jit
    def _publish_phase_37(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 37)
    @cute.jit
    def _publish_phase_38(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 38)
    @cute.jit
    def _publish_phase_40(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 40)
    @cute.jit
    def _publish_phase_41(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 41)
    @cute.jit
    def _publish_phase_42(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 42)
    @cute.jit
    def _publish_phase_43(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 43)
    @cute.jit
    def _publish_phase_44(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 44)
    @cute.jit
    def _publish_phase_47(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 47)
    @cute.jit
    def _publish_phase_49(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 49)
    @cute.jit
    def _publish_phase_51(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 51)
    @cute.jit
    def _publish_phase_52(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 52)
    @cute.jit
    def _publish_phase_53(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 53)
    @cute.jit
    def _publish_phase_54(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 54)
    @cute.jit
    def _publish_phase_55(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 55)
    @cute.jit
    def _publish_phase_56(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 56)
    @cute.jit
    def _publish_phase_58(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 58)
    @cute.jit
    def _publish_phase_59(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 59)
    @cute.jit
    def _publish_phase_60(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 60)
    @cute.jit
    def _publish_phase_61(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 61)
    @cute.jit
    def _publish_phase_62(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 62)
    @cute.jit
    def _publish_phase_65(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 65)
    @cute.jit
    def _publish_phase_67(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 67)
    @cute.jit
    def _publish_phase_69(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 69)
    @cute.jit
    def _publish_phase_70(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 70)
    @cute.jit
    def _publish_phase_71(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 71)
    @cute.jit
    def _publish_phase_72(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 72)
    @cute.jit
    def _publish_phase_73(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 73)
    @cute.jit
    def _publish_phase_74(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 74)
    @cute.jit
    def _publish_phase_76(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 76)
    @cute.jit
    def _publish_phase_77(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 77)
    @cute.jit
    def _publish_phase_78(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 78)
    @cute.jit
    def _publish_phase_79(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 79)
    @cute.jit
    def _publish_phase_80(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 80)
    @cute.jit
    def _publish_phase_83(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 83)
    @cute.jit
    def _publish_phase_85(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 85)
    @cute.jit
    def _publish_phase_87(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 87)
    @cute.jit
    def _publish_phase_88(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 88)
    @cute.jit
    def _publish_phase_89(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 89)
    @cute.jit
    def _publish_phase_90(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 90)
    @cute.jit
    def _publish_phase_91(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 91)
    @cute.jit
    def _publish_phase_92(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 92)
    @cute.jit
    def _publish_phase_94(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 94)
    @cute.jit
    def _publish_phase_95(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 95)
    @cute.jit
    def _publish_phase_96(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 96)
    @cute.jit
    def _publish_phase_97(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 97)
    @cute.jit
    def _publish_phase_98(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 98)
    @cute.jit
    def _publish_phase_101(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 101)
    @cute.jit
    def _publish_phase_103(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 103)
    @cute.jit
    def _publish_phase_105(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 105)
    @cute.jit
    def _publish_phase_106(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 106)
    @cute.jit
    def _publish_phase_107(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 107)
    @cute.jit
    def _publish_phase_108(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 108)
    @cute.jit
    def _publish_phase_109(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 109)
    @cute.jit
    def _publish_phase_110(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 110)
    @cute.jit
    def _publish_phase_112(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 112)
    @cute.jit
    def _publish_phase_113(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 113)
    @cute.jit
    def _publish_phase_114(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 114)
    @cute.jit
    def _publish_phase_115(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 115)
    @cute.jit
    def _publish_phase_116(epoch: cutlass.Int32):
        grid_barrier(ready, workers, epoch + 116)

    cooperative_worker_loop(
        ready,
        workers,
        epoch,
        (
            (0, _run_variant_0, _no_op, _cta_sync),
            (1, _run_variant_0, _no_op, _cta_sync),
            (2, _run_variant_0, _no_op, _cta_sync),
            (3, _run_variant_1, _no_op, _cta_sync),
            (4, _run_variant_2, _no_op, _cta_sync),
            (5, _run_variant_3, _no_op, _cta_sync),
            (6, _run_variant_3, _no_op, _cta_sync),
            (7, _run_variant_4, _no_op, _cta_sync),
            (8, _run_variant_5, _no_op, _cta_sync),
            (9, _run_variant_6, _no_op, _publish_phase_9),
            (10, _run_variant_7, _no_op, _cta_sync),
            (11, _run_variant_8, _no_op, _cta_sync),
            (12, _run_variant_9, _no_op, _cta_sync),
            (13, _run_variant_10, _no_op, _cta_sync),
            (14, _run_variant_11, _no_op, _cta_sync),
            (15, _run_variant_12, _no_op, _publish_phase_15),
            (16, _run_variant_13, _no_op, _cta_sync),
            (17, _run_variant_14, _no_op, _cta_sync),
            (18, _run_variant_13, _no_op, _cta_sync),
            (19, _run_variant_15, _no_op, _cta_sync),
            (20, _run_variant_13, _no_op, _publish_phase_20),
            (21, _run_variant_16, _no_op, _cta_sync),
            (22, _run_variant_17, _no_op, _cta_sync),
            (23, _run_variant_18, _no_op, _cta_sync),
            (24, _run_variant_19, _no_op, _publish_phase_24),
            (25, _run_variant_20, _no_op, _cta_sync),
            (26, _run_variant_21, _no_op, _cta_sync),
            (27, _run_variant_22, _no_op, _publish_phase_27),
            (28, _run_variant_23, _no_op, _cta_sync),
            (29, _run_variant_24, _no_op, _cta_sync),
            (30, _run_variant_25, _no_op, _publish_phase_30),
            (31, _run_variant_22, _no_op, _publish_phase_31),
            (32, _run_variant_26, _no_op, _publish_phase_32),
            (33, _run_variant_27, _no_op, _cta_sync),
            (34, _run_variant_28, _no_op, _publish_phase_34),
            (35, _run_variant_29, _no_op, _publish_phase_35),
            (36, _run_variant_13, _no_op, _publish_phase_36),
            (37, _run_variant_30, _no_op, _publish_phase_37),
            (38, _run_variant_7, _no_op, _publish_phase_38),
            (39, _run_variant_13, _no_op, _cta_sync),
            (40, _run_variant_13, _no_op, _publish_phase_40),
            (41, _run_variant_31, _no_op, _publish_phase_41),
            (42, _run_variant_13, _no_op, _publish_phase_42),
            (43, _run_variant_30, _no_op, _publish_phase_43),
            (44, _run_variant_7, _no_op, _publish_phase_44),
            (45, _run_variant_13, _no_op, _cta_sync),
            (46, _run_variant_13, _no_op, _cta_sync),
            (47, _run_variant_13, _no_op, _publish_phase_47),
            (48, _run_variant_16, _no_op, _cta_sync),
            (49, _run_variant_18, _no_op, _publish_phase_49),
            (50, _run_variant_23, _no_op, _cta_sync),
            (51, _run_variant_24, _no_op, _publish_phase_51),
            (52, _run_variant_28, _no_op, _publish_phase_52),
            (53, _run_variant_29, _no_op, _publish_phase_53),
            (54, _run_variant_13, _no_op, _publish_phase_54),
            (55, _run_variant_30, _no_op, _publish_phase_55),
            (56, _run_variant_7, _no_op, _publish_phase_56),
            (57, _run_variant_13, _no_op, _cta_sync),
            (58, _run_variant_13, _no_op, _publish_phase_58),
            (59, _run_variant_31, _no_op, _publish_phase_59),
            (60, _run_variant_13, _no_op, _publish_phase_60),
            (61, _run_variant_30, _no_op, _publish_phase_61),
            (62, _run_variant_7, _no_op, _publish_phase_62),
            (63, _run_variant_13, _no_op, _cta_sync),
            (64, _run_variant_13, _no_op, _cta_sync),
            (65, _run_variant_13, _no_op, _publish_phase_65),
            (66, _run_variant_16, _no_op, _cta_sync),
            (67, _run_variant_18, _no_op, _publish_phase_67),
            (68, _run_variant_23, _no_op, _cta_sync),
            (69, _run_variant_24, _no_op, _publish_phase_69),
            (70, _run_variant_28, _no_op, _publish_phase_70),
            (71, _run_variant_29, _no_op, _publish_phase_71),
            (72, _run_variant_13, _no_op, _publish_phase_72),
            (73, _run_variant_30, _no_op, _publish_phase_73),
            (74, _run_variant_7, _no_op, _publish_phase_74),
            (75, _run_variant_13, _no_op, _cta_sync),
            (76, _run_variant_13, _no_op, _publish_phase_76),
            (77, _run_variant_31, _no_op, _publish_phase_77),
            (78, _run_variant_13, _no_op, _publish_phase_78),
            (79, _run_variant_30, _no_op, _publish_phase_79),
            (80, _run_variant_7, _no_op, _publish_phase_80),
            (81, _run_variant_13, _no_op, _cta_sync),
            (82, _run_variant_13, _no_op, _cta_sync),
            (83, _run_variant_13, _no_op, _publish_phase_83),
            (84, _run_variant_16, _no_op, _cta_sync),
            (85, _run_variant_18, _no_op, _publish_phase_85),
            (86, _run_variant_23, _no_op, _cta_sync),
            (87, _run_variant_24, _no_op, _publish_phase_87),
            (88, _run_variant_28, _no_op, _publish_phase_88),
            (89, _run_variant_29, _no_op, _publish_phase_89),
            (90, _run_variant_13, _no_op, _publish_phase_90),
            (91, _run_variant_30, _no_op, _publish_phase_91),
            (92, _run_variant_7, _no_op, _publish_phase_92),
            (93, _run_variant_13, _no_op, _cta_sync),
            (94, _run_variant_13, _no_op, _publish_phase_94),
            (95, _run_variant_31, _no_op, _publish_phase_95),
            (96, _run_variant_13, _no_op, _publish_phase_96),
            (97, _run_variant_30, _no_op, _publish_phase_97),
            (98, _run_variant_7, _no_op, _publish_phase_98),
            (99, _run_variant_13, _no_op, _cta_sync),
            (100, _run_variant_13, _no_op, _cta_sync),
            (101, _run_variant_13, _no_op, _publish_phase_101),
            (102, _run_variant_16, _no_op, _cta_sync),
            (103, _run_variant_18, _no_op, _publish_phase_103),
            (104, _run_variant_23, _no_op, _cta_sync),
            (105, _run_variant_24, _no_op, _publish_phase_105),
            (106, _run_variant_28, _no_op, _publish_phase_106),
            (107, _run_variant_29, _no_op, _publish_phase_107),
            (108, _run_variant_13, _no_op, _publish_phase_108),
            (109, _run_variant_30, _no_op, _publish_phase_109),
            (110, _run_variant_7, _no_op, _publish_phase_110),
            (111, _run_variant_13, _no_op, _cta_sync),
            (112, _run_variant_13, _no_op, _publish_phase_112),
            (113, _run_variant_31, _no_op, _publish_phase_113),
            (114, _run_variant_13, _no_op, _publish_phase_114),
            (115, _run_variant_30, _no_op, _publish_phase_115),
            (116, _run_variant_7, _no_op, _publish_phase_116),
            (117, _run_variant_13, _no_op, _ignore_epoch),
            (118, _run_terminal, _no_op, _ignore_epoch)
        ),
    )



@cute.jit
def launch(
    ready: cute.Tensor,
    workers: cutlass.Int32,
    epoch: cutlass.Int32,
    work_iterators: tuple[cute.Tensor, ...],
    tensors: tuple[cute.Tensor, ...],
    g0_tensors: tuple[tuple[cute.Tensor, cute.Tensor, cute.Tensor], ...],
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream
):
    g0_values = (
        _g0_body_13.prepare(
            g0_tensors[0][0],
            g0_tensors[0][1],
            g0_tensors[0][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[1][0],
            g0_tensors[1][1],
            g0_tensors[1][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[2][0],
            g0_tensors[2][1],
            g0_tensors[2][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[3][0],
            g0_tensors[3][1],
            g0_tensors[3][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[4][0],
            g0_tensors[4][1],
            g0_tensors[4][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[5][0],
            g0_tensors[5][1],
            g0_tensors[5][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[6][0],
            g0_tensors[6][1],
            g0_tensors[6][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[7][0],
            g0_tensors[7][1],
            g0_tensors[7][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[8][0],
            g0_tensors[8][1],
            g0_tensors[8][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[9][0],
            g0_tensors[9][1],
            g0_tensors[9][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[10][0],
            g0_tensors[10][1],
            g0_tensors[10][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[11][0],
            g0_tensors[11][1],
            g0_tensors[11][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[12][0],
            g0_tensors[12][1],
            g0_tensors[12][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[13][0],
            g0_tensors[13][1],
            g0_tensors[13][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[14][0],
            g0_tensors[14][1],
            g0_tensors[14][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[15][0],
            g0_tensors[15][1],
            g0_tensors[15][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[16][0],
            g0_tensors[16][1],
            g0_tensors[16][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[17][0],
            g0_tensors[17][1],
            g0_tensors[17][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[18][0],
            g0_tensors[18][1],
            g0_tensors[18][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[19][0],
            g0_tensors[19][1],
            g0_tensors[19][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[20][0],
            g0_tensors[20][1],
            g0_tensors[20][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[21][0],
            g0_tensors[21][1],
            g0_tensors[21][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[22][0],
            g0_tensors[22][1],
            g0_tensors[22][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[23][0],
            g0_tensors[23][1],
            g0_tensors[23][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[24][0],
            g0_tensors[24][1],
            g0_tensors[24][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[25][0],
            g0_tensors[25][1],
            g0_tensors[25][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[26][0],
            g0_tensors[26][1],
            g0_tensors[26][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[27][0],
            g0_tensors[27][1],
            g0_tensors[27][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[28][0],
            g0_tensors[28][1],
            g0_tensors[28][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[29][0],
            g0_tensors[29][1],
            g0_tensors[29][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[30][0],
            g0_tensors[30][1],
            g0_tensors[30][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[31][0],
            g0_tensors[31][1],
            g0_tensors[31][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[32][0],
            g0_tensors[32][1],
            g0_tensors[32][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[33][0],
            g0_tensors[33][1],
            g0_tensors[33][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[34][0],
            g0_tensors[34][1],
            g0_tensors[34][2],
            max_active_clusters,
        ),
        _g0_body_13.prepare(
            g0_tensors[35][0],
            g0_tensors[35][1],
            g0_tensors[35][2],
            max_active_clusters,
        ),
    )
    megabake_kernel(
        ready,
        workers,
        epoch,
        work_iterators,
        tensors,
        g0_values
    ).launch(
        grid=[1, 1, workers],
        block=[256, 1, 1],
        cooperative=True,
        stream=stream,
    )

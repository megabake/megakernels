#!/bin/bash
# Chunk 3: attention scores in shared memory, direct-store GEMM epilogue,
# working L2 weight prefetch (budget sweep), then the GPU test suite.
cd /workspace/megabake
OUT=artifacts/chunk3_2026-10-20
for model in HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B Qwen/Qwen2-0.5B Qwen/Qwen3.5-0.8B; do
  slug=${model//\//--}
  for budget in 0 8388608 25165824; do
    MEGABAKE_L2_PREFETCH_BYTES=$budget .venv/bin/python benchmarks/full_model.py --modes decode --variants megabake \
      --model $model --out $OUT/decode_pf$budget/$slug >> $OUT/decode.log 2>&1
  done
done
.venv/bin/python benchmarks/full_model.py --modes decode --variants inductor_graphed \
  --model Qwen/Qwen3.5-0.8B --out $OUT/decode_graphed/Qwen--Qwen3.5-0.8B >> $OUT/decode.log 2>&1
HF_HUB_OFFLINE=1 timeout 5400 .venv/bin/python -m pytest tests/gpu -q -p no:cacheprovider > $OUT/gpu_tests.log 2>&1
echo "tests exit $?" >> $OUT/gpu_tests.log
echo done > $OUT/DONE

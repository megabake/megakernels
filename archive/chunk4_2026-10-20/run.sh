#!/bin/bash
# Chunk 4: one-load slot addresses, affine iterators, next-row L1 prefetch,
# idle-CTA L2 prefetch (on/off), then the GPU test suite.
cd /workspace/megabake
OUT=artifacts/chunk4_2026-10-20
for model in HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B Qwen/Qwen2-0.5B Qwen/Qwen3.5-0.8B; do
  slug=${model//\//--}
  for budget in 0 8388608; do
    MEGABAKE_L2_PREFETCH_BYTES=$budget .venv/bin/python benchmarks/full_model.py --modes decode --variants megabake \
      --model $model --out $OUT/decode_pf$budget/$slug >> $OUT/decode.log 2>&1
  done
done
HF_HUB_OFFLINE=1 timeout 5400 .venv/bin/python -m pytest tests/gpu -q -p no:cacheprovider > $OUT/gpu_tests.log 2>&1
echo "tests exit $?" >> $OUT/gpu_tests.log
echo done > $OUT/DONE

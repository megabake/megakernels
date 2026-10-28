#!/bin/bash
# Step 1 and 2 completion: CSE, shared GEMM variants with run-time epilogue kinds,
# lazy epilogue state, predicated direct stores. GPU tests, then all 8 models against both Inductor
# baselines, then per-CTA phase profiles of 135M and 1.7B decode.
cd /workspace/megabake
OUT=artifacts/step12b_2026-10-21
MEGABAKE_RUN_GPU_TESTS=1 HF_HUB_OFFLINE=1 timeout 7200 .venv/bin/python -m pytest tests/gpu -q -p no:cacheprovider -W ignore > $OUT/gpu_tests.log 2>&1
echo "tests exit $?" >> $OUT/gpu_tests.log
for model in HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B Qwen/Qwen2-0.5B Qwen/Qwen3.5-0.8B \
    jukofyork/DeepSeek-V3-DRAFT-0.6B-v3.0 reflection/Beam-1.0-501B-A23B-DFlash2 google/gemma-3-270m google/gemma-2b; do
  .venv/bin/python benchmarks/full_model.py --modes prefill decode \
    --variants inductor_cudagraphs inductor_graphed megabake \
    --model $model --out $OUT/full_model/${model//\//--} >> $OUT/run.log 2>&1
  echo "$model exit $?" >> $OUT/status.log
done
.venv/bin/python tools/phase_profile.py --mode decode --cta --out $OUT/phases_135m.json > $OUT/phases_135m.log 2>&1
.venv/bin/python tools/phase_profile.py --model HuggingFaceTB/SmolLM-1.7B --mode decode --cta --out $OUT/phases_1.7b.json > $OUT/phases_1.7b.log 2>&1
echo done > $OUT/DONE

#!/bin/bash
# Step 3: weight streaming into shared memory (default on). GPU tests, C7519
# status of every kernel, then all 8 models against both Inductor baselines.
cd /workspace/megabake
OUT=artifacts/step3_2026-10-21
MEGABAKE_RUN_GPU_TESTS=1 HF_HUB_OFFLINE=1 timeout 7200 .venv/bin/python -m pytest tests/gpu -q -p no:cacheprovider -W ignore > $OUT/gpu_tests.log 2>&1
echo "tests exit $?" >> $OUT/gpu_tests.log
MODELS="HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B Qwen/Qwen2-0.5B Qwen/Qwen3.5-0.8B \
    jukofyork/DeepSeek-V3-DRAFT-0.6B-v3.0 reflection/Beam-1.0-501B-A23B-DFlash2 google/gemma-3-270m google/gemma-2b"
for model in $MODELS; do
  for mode in prefill decode; do
    [ $model = Qwen/Qwen3.5-0.8B ] && [ $mode = prefill ] && continue
    HF_HUB_OFFLINE=1 .venv/bin/python tools/wgmma_serialization.py --model $model --mode $mode 2>&1 \
      | grep -E "wgmma (not )?serialized" >> $OUT/wgmma.log || echo "$model $mode: check failed" >> $OUT/wgmma.log
  done
done
for model in $MODELS; do
  HF_HUB_OFFLINE=1 .venv/bin/python benchmarks/full_model.py --modes prefill decode \
    --variants inductor_cudagraphs inductor_graphed megabake \
    --model $model --out $OUT/full_model/${model//\//--} >> $OUT/run.log 2>&1
  echo "$model exit $?" >> $OUT/status.log
done
echo done > $OUT/DONE

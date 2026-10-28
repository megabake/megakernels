#!/bin/bash
# Plain Inductor (no CUDA graphs) and eager for all 8 models, to complete
# the step12b_2026-10-21 comparison.
cd /workspace/megabake
OUT=artifacts/inductor_eager_2026-10-21
for model in HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B Qwen/Qwen2-0.5B Qwen/Qwen3.5-0.8B \
    jukofyork/DeepSeek-V3-DRAFT-0.6B-v3.0 reflection/Beam-1.0-501B-A23B-DFlash2 google/gemma-3-270m google/gemma-2b; do
  .venv/bin/python benchmarks/full_model.py --modes prefill decode --variants eager inductor \
    --model $model --out $OUT/${model//\//--} >> $OUT/run.log 2>&1
  echo "$model exit $?" >> $OUT/status.log
done
echo done > $OUT/DONE

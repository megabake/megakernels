#!/bin/bash
# Chunk 2: idle-phase skipping plus an L2 weight-prefetch budget sweep.
cd /workspace/megabake
OUT=artifacts/chunk2_2026-10-20
export HF_HUB_OFFLINE=1
for model in HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B Qwen/Qwen2-0.5B google/gemma-3-270m; do
  slug=${model//\//--}
  for budget in 0 8388608 25165824; do
    MEGABAKE_L2_PREFETCH_BYTES=$budget .venv/bin/python benchmarks/full_model.py --modes decode --variants megabake \
      --model $model --out $OUT/decode_pf$budget/$slug >> $OUT/decode.log 2>&1
  done
done
.venv/bin/python benchmarks/full_model.py --modes decode --variants inductor_graphed \
  --model google/gemma-3-270m --out $OUT/decode_graphed/google--gemma-3-270m >> $OUT/decode.log 2>&1
echo done > $OUT/DONE

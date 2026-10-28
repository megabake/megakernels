#!/bin/bash
# Phase 0 measurements: per-CTA decode profiles and the graphed decode baseline.
cd /workspace/megabake
OUT=artifacts/phase0_2026-10-20
.venv/bin/python tools/phase_profile.py --mode decode --cta --calls 9 --out $OUT/profile_135m.json > $OUT/profile_135m.log 2>&1
.venv/bin/python tools/phase_profile.py --model HuggingFaceTB/SmolLM-1.7B --mode decode --cta --calls 9 --out $OUT/profile_1.7b.json > $OUT/profile_1.7b.log 2>&1
.venv/bin/python benchmarks/full_model.py --modes decode --variants inductor inductor_graphed megabake \
  --model HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM-1.7B Qwen/Qwen2-0.5B google/gemma-3-270m Qwen/Qwen3.5-0.8B \
  --out $OUT/decode > $OUT/decode.log 2>&1
echo done > $OUT/DONE

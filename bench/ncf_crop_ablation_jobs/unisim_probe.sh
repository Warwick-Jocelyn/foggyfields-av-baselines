#!/bin/bash
# Short UniSim compatibility probe: install plugin if needed, then a tiny train on NCF-02.
source /capstor/scratch/cscs/yitiwang/ncf-crop-ablation/jobs/env.sh
python -c "import unisim" 2>/dev/null || pip install -q --no-deps --no-build-isolation -e "$CODE/_src/unisim"
python -c "from unisim.unisim_config import unisim; print('import OK')" || exit 3
OUT=$W/outputs_unisim_probe; rm -rf "$OUT"
cd "$NS_REPO" && timeout 1100 ns-train unisim $UNISIM_EXTRA --output-dir "$OUT" --vis tensorboard \
  --max-num-iterations ${PROBE_ITERS:-30} --steps-per-save ${PROBE_ITERS:-30} --machine.seed 42 \
  --steps-per-eval-image 999999 --steps-per-eval-batch 999999 --steps-per-eval-all-images 999999 \
  --viewer.quit-on-train-completion True \
  nvidia-av-fog-data --data "$W/data" --sequence NCF-02 --rectification v2 --splits-json nvs_50_50 \
    --apply-safe-crop True --use-valid-masks False $UNISIM_DP_EXTRA --dataset-end-fraction 0.1 \
    --cameras camera_front_wide_120fov camera_rear_left_70fov camera_front_tele_30fov
echo "PROBE rc=$?"

# NeuRAD diagnosis (blurry actors/statics, dark blobs on thin objects): variant -> extra model flags.
# Every variant keeps use-temporal-appearance False (= neurad-paper). flipoff also fixes the preset bug:
# method_configs["neurad-paper"] sets `f.flip_prob = 0.0`, an attribute nothing reads, so actor
# mirroring stayed on (grid.actor.flip_prob 0.25 / 0.5).
NEURAD_BASE="--pipeline.model.use-temporal-appearance False"
FLIPOFF="--pipeline.model.field.grid.actor.flip-prob 0 --pipeline.model.sampling.proposal-field-1.grid.actor.flip-prob 0 --pipeline.model.sampling.proposal-field-2.grid.actor.flip-prob 0"
BIGGRID="--pipeline.model.field.grid.static.log2-hashmap-size 23 --pipeline.model.field.grid.static.max-res 16384 --pipeline.model.field.grid.actor.log2-hashmap-size 20 --pipeline.model.field.grid.actor.max-res 2048"
UP2="--pipeline.model.rgb-upsample-factor 2"
variant_flags(){
  case "$1" in
    flipoff) echo "$NEURAD_BASE $FLIPOFF" ;;
    biggrid) echo "$NEURAD_BASE $BIGGRID" ;;
    up2)     echo "$NEURAD_BASE $UP2" ;;
    all)     echo "$NEURAD_BASE $FLIPOFF $BIGGRID $UP2" ;;
    posefix)    echo "$NEURAD_BASE" ;;
    posefix_fb) echo "$NEURAD_BASE $FLIPOFF $BIGGRID" ;;
    boxfix|boxpose) echo "$NEURAD_BASE" ;;
    boxpose_fb) echo "$NEURAD_BASE $FLIPOFF $BIGGRID" ;;
    *) return 1 ;;
  esac
}
# Dataparser flags; empty = env.sh default (--add-missing-points True, as in the bench). posefix*: camera
# poses at the camera's own exposure time instead of the LiDAR spin centre (F1b, 12-17 cm at 11 m/s).
variant_dp_flags(){
  case "$1" in
    posefix|posefix_fb) echo "--add-missing-points True --camera-pose-time camera" ;;
    boxfix) echo "--add-missing-points True --actor-box-axes pandaset" ;;   # F2b: boxes were rotated 90 deg
    boxpose|boxpose_fb) echo "--add-missing-points True --actor-box-axes pandaset --camera-pose-time camera" ;;
    *) echo "" ;;
  esac
}

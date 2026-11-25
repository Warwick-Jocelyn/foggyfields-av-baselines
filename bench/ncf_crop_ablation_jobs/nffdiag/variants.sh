# SplatAD-on-fog diagnosis: variant name -> extra ns-train model flags (appended to lane.sh's SplatAD flags).
# Background: on NFF the mean Gaussian opacity collapses (~0.0006 vs ~0.1 on NCF) and MCMC's
# opacity-dependent position noise then smears every Gaussian -> grey blobs. Each variant removes one
# suspected driver of that collapse.
variant_flags(){
  case "$1" in
    base)      echo "" ;;
    los0)      echo "--pipeline.model.line-of-sight-lambda 0" ;;
    lidarlite) echo "--pipeline.model.line-of-sight-lambda 0 --pipeline.model.depth-lambda 0.01" ;;
    camonly)   echo "--pipeline.model.line-of-sight-lambda 0 --pipeline.model.depth-lambda 0 --pipeline.model.intensity-lambda 0 --pipeline.model.ray-drop-lambda 0" ;;
    opreg0)    echo "--pipeline.model.mcmc-opacity-reg-lambda 0" ;;
    opreg5e4)  echo "--pipeline.model.mcmc-opacity-reg-lambda 0.0005" ;;
    noise)     echo "--pipeline.model.mcmc-noise-lr 50000" ;;
    camopt)    echo "--pipeline.model.camera-optimizer.mode SO3xR3" ;;
    bgblack)   echo "--pipeline.model.background-color black" ;;
    posefix|boxfix|boxpose) echo "" ;;
    allfix)    echo "--pipeline.model.mcmc-opacity-reg-lambda 0.0005" ;;
    *) return 1 ;;
  esac
}
# Dataparser flags (after the nvidia-av-fog-data subcommand). posefix: camera poses at the camera's own
# exposure time instead of the LiDAR spin centre (F1b, 12-17 cm at 11 m/s).
variant_dp_flags(){
  case "$1" in
    posefix) echo "--camera-pose-time camera" ;;
    boxfix)  echo "--actor-box-axes pandaset" ;;   # F2b: actor boxes were rotated 90 deg
    boxpose|allfix) echo "--actor-box-axes pandaset --camera-pose-time camera" ;;
    *) echo "" ;;
  esac
}

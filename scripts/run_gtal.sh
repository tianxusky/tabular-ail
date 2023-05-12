ENV=CliffWalking
DIST_TYPE=NonUniform
NUM_TRAJ=100

if [ "$(uname)" == "Darwin" ]; then
  python -m gtal.main -s \
  env.id=${ENV} \
  env.init_dist_type=${DIST_TYPE} \
  env.num_traj=${NUM_TRAJ} \
  seed=100 \
  algorithm="GTAL"

elif [ "$(uname)" == "Linux" ]; then
  for SEED in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
  do
      python -m gtal.main -s \
      env.id=${ENV} \
      env.init_dist_type=${DIST_TYPE} \
      env.num_traj=${NUM_TRAJ} \
      seed=${SEED} \
      algorithm="GTAL" & sleep 2
  done
  wait
fi


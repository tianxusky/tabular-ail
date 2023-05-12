ENV=CliffWalking
DIST_TYPE=NonUniform
NUM_traj=100

if [ "$(uname)" == "Darwin" ]; then
  python -m fem.main -s \
  env.id=${ENV} \
  env.init_dist_type=${DIST_TYPE} \
  env.num_traj=${NUM_traj} \
  seed=100 \
  algorithm="FEM"

elif [ "$(uname)" == "Linux" ]; then
  for SEED in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
  do
      python -m fem.main -s \
      env.id=${ENV} \
      env.num_traj=${NUM_traj} \
      env.init_dist_type=${DIST_TYPE} \
      seed=${SEED} \
      algorithm="FEM" & sleep 2
  done
  wait
fi


#!/bin/bash

# This file runs all experiments as used in the ICAPS23 paper, and stores them in the data-folder. For creating plots, see the Plot_Data.ipynb
# Note: in reality, all data has been gathered by manually starting runs. Thus, this code is meant more as a reference and might take a very long time to run!

# Experiments on small frozen lake, for different variants:
echo -e "\n\n============= Alpha_real changing, 12x12 Frozen Lake =============\n\n"
eps=1000
runs=1
for alpha_plan in 0.6 0.7 0.8 #$(seq 0.6 0.1 0.8001)
do
    for alpha_real in $(seq 0.55 0.05 1.001)
    do
        python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12
        python3 ./Run.py -alg ATM_RMDP              -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12
        # python3 ./Run.py -alg ATM_Robust            -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12
        # python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 0    -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12
        # python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 1    -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12
        # python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure -1   -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12
        # Note: alpha_measure = alpha_plan!
    done
done

# echo -e "\n\n============= Alpha_real changing, Maintenace  =============\n\n"
# eps=1000
# runs=1
# for alpha_plan in $(seq 0.9 1 0.91)
# do
#     for alpha_real in $(seq 0.8 0.01 1.001)
#     do
#         python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Maintenance -nmbr_eps $eps -nmbr_runs $runs
#         python3 ./Run.py -alg ATM_RMDP              -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Maintenance -nmbr_eps $eps -nmbr_runs $runs
#         python3 ./Run.py -alg ATM_Robust            -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Maintenance -nmbr_eps $eps -nmbr_runs $runs
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 0    -env Maintenance -nmbr_eps $eps -nmbr_runs $runs        
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 1    -env Maintenance -nmbr_eps $eps -nmbr_runs $runs
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure -1   -env Maintenance -nmbr_eps $eps -nmbr_runs $runs
#         # Note: alpha_measure = alpha_plan!
#     done
# done


echo -e "\n\n============= RUNS COMPLETED =============\n\n"
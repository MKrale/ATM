#!/bin/bash

# This file runs all experiments as used in the ICAPS23 paper, and stores them in the data-folder. For creating plots, see the Plot_Data.ipynb
# Note: in reality, all data has been gathered by manually starting runs. Thus, this code is meant more as a reference and might take a very long time to run!

# Experiments on small frozen lake, for different variants:
echo -e "\n\n============= Alpha_real changing, 12x12 Frozen Lake =============\n\n"
eps=1000
runs=1
i=0
for alpha_plan in $(seq 0.6 0.1 0.8001)
do
    for alpha_real in $(seq 0.55 0.1 1.001)
    do
        python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 &
        i=$i+1
        if (($i==6))
        then
            i=0
            wait
        fi 
    done
done

for alpha_plan in $(seq 0.6 0.1 0.8001)
do
    for alpha_real in $(seq 0.55 0.1 1.001)
    do
        python3 ./Run.py -alg ATM_RMDP              -alpha_real $alpha_real -alpha_plan $alpha_plan                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
        python3 ./Run.py -alg ATM_Robust            -alpha_real $alpha_real -alpha_plan $alpha_plan                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure $alpha_real  -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 1            -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure -$alpha_real -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
        wait
        # Note: alpha_measure = alpha_plan!
    done
done
wait





echo -e "\n\n============= Alpha_real changing, Maintenace  =============\n\n"
eps=1000
runs=1
i=0
for alpha_plan in $(seq 0.8 0.05 0.91)
do
    for alpha_real in $(seq 0.8 0.01 1.001)
    do
        python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Maintenance -nmbr_eps $eps -nmbr_runs $runs &
        i=$i+1
        if (($i==6))
        then
            i=0
            wait
        fi 
    done
done

for alpha_plan in $(seq 0.8 0.05 0.91)
do
    for alpha_real in $(seq 0.8 0.01 1.001)
    do
        
        wait
        python3 ./Run.py -alg ATM_RMDP              -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Maintenance -nmbr_eps $eps -nmbr_runs $runs &
        python3 ./Run.py -alg ATM_Robust            -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Maintenance -nmbr_eps $eps -nmbr_runs $runs &
        python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 0    -env Maintenance -nmbr_eps $eps -nmbr_runs $runs &      
        python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 1    -env Maintenance -nmbr_eps $eps -nmbr_runs $runs &
        python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure -1   -env Maintenance -nmbr_eps $eps -nmbr_runs $runs &
        # Note: alpha_measure = alpha_plan!
    done
done
wait




echo -e "\n\n============= Alpha_real changing, uMV  =============\n\n"
eps=1000
runs=1
i=0
folder_path="Data/Temp1/"
# get base environment
for p_plan in 0.1 0.2 0.3 0.7 0.8 1
do
    for p_real in $(seq 0.05 0.05 1)
    do
        p_plan_set=$( python3 -c "print($p_plan/2)") 
        python3 ./Run.py -alg ATM  -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        if (($i==6))
        then
            i=0
            wait
        fi 
    done
done


for p_plan in 0.1 0.2 0.3 0.7 0.8 1
do
    for p_real in $(seq 0.05 0.05 1)
    do
        p_plan_set=$( python3 -c "print($p_plan/2)")
        python3 ./Run.py -alg ATM_Robust            -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_RMDP              -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 1 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure -0.5 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        # Note: alpha_measure = alpha_plan!
    done
done
wait

echo -e "\n\n============= RUNS COMPLETED =============\n\n"
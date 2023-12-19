#!/bin/bash

#       This file runs all experiments as used in the paper, and stores them in the data-folder. For creating plots, see the Plot_Data.ipynb
#       Note: this file has been written to efficiently use all resources on our setup. 
#       Other setups may require altering the code such that less (or more) runs occur at once.
#       Furthermore, in running all code in this file may take a long time: in practice we ran our experiments in stages.

nmbr_cores = 5
# nmbr_cores=12

##########################################################
#        Snake Maze Environment:
##########################################################

echo -e "\n\n============= Snake Maze, changing alpha  =============\n\n"

folder_path="Data/Robust_Results/SnakeMaze/"
eps=50
runs=1
i=0
# Get certain env:
python3 ./Run.py -alg ATM -env SnakeMaze -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path

for alpha_plan in 0.6
do
    alpha_plan=1
    for alpha_real in $(seq 0.55 0.01 1.01)
    do
        echo -e "Alpha_plan = $alpha_plan , Alpha_real = $alpha_real "
        python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        i=$((i+1))
        if (("$i"=="$nmbr_cores"));
        then
            i=0
            wait
        fi 
    done
done
wait
i=0
for alpha_plan in 0.6
do
    for alpha_real in $(seq 0.55 0.01 1.01)
    do
        echo -e "Alpha_plan = $alpha_plan , Alpha_real = $alpha_real, cost = $mcost "
        python3 ./Run.py -alg ATM                        -alpha_real $alpha_real -alpha_plan $alpha_plan                                -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_RMDP                   -alpha_real $alpha_real -alpha_plan $alpha_plan                                -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_Robust                 -alpha_real $alpha_real -alpha_plan $alpha_plan                                -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        ### Pessimistic
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure $alpha_plan     -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        ### Average
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 1               -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        ### Optimistic
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure -$alpha_plan    -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        i=$((i+6))
        inext=$((i+6))
        if (("$inext">"$nmbr_cores"));
        then
            i=0
            wait
        fi 
    done
    wait
done
wait


echo -e "\n\n============= Snake Maze, constant alpha  =============\n\n"

folder_path="Data/Robust_Results/SnakeMaze/"
eps=50
runs=1
i=0
# Get certain env:
python3 ./Run.py -alg ATM -env SnakeMaze -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path

for alpha_plan in 0.6
do
    alpha_plan=1
    for alpha_real in $(seq 0.55 0.05 1.01)
    do
        echo -e "Alpha_plan = $alpha_plan , Alpha_real = $alpha_real "
        python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        i=$((i+1))
        if (("$i"=="$nmbr_cores"));
        then
            i=0
            wait
        fi 
    done
done
wait
i=0
for alpha_real in $(seq 0.55 0.01 0.65)
do
    echo -e "Alpha_real = $alpha_real, cost = $mcost "
    python3 ./Run.py -alg ATM                        -alpha_real $alpha_real -alpha_plan $alpha_real                                -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    python3 ./Run.py -alg ATM_RMDP                   -alpha_real $alpha_real -alpha_plan $alpha_real                                -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    python3 ./Run.py -alg ATM_Robust                 -alpha_real $alpha_real -alpha_plan $alpha_real                                -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    ### Pessimistic
    python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure $alpha_real     -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    ### Average
    python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure 1               -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    ### Optimistic
    python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure -$alpha_real    -env SnakeMaze  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    i=$((i+3))
    inext=$((i+3))
    if (("$inext">"$nmbr_cores"));
    then
        i=0
        wait
    fi 
done
wait




##########################################################
#        Lucky-Unlucky Environment:
##########################################################

echo -e "\n\n============= Alpha_real changing, uMV  =============\n\n"
eps=1000
runs=1
i=0
folder_path="Data/Robust_Results/uMV/"
# get base environment
for p_plan in 0.1 0.2 0.3 0.7 0.8 1
do
    for p_real in $(seq 0.05 0.05 1)
    do
        p_plan_set=$( python3 -c "print($p_plan/2)") 
        python3 ./Run.py -alg ATM  -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        i=$((i+1))
        if (("$i"=="$nmbr_cores"));
        then
            i=0
            wait
        fi 
    done
done
wait


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
    done
done


# ##########################################################
# #        A-B Environment:
# ##########################################################


echo -e "\n\n============= Alpha_plan changing, uMV  =============\n\n"
eps=1000
runs=1
i=0
folder_path="Data/Robust_Results/uMV/"
# get base environment
for p_plan in $(seq 0.05 0.01 1)
do
    for p_real in 0.5
    do
        p_plan_set=$( python3 -c "print($p_plan/2)") 
        python3 ./Run.py -alg ATM  -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        i=$((i+1))
        if (("$i"=="$nmbr_cores"));
        then
            i=0
            wait
        fi 
    done
done
wait
for p_plan in $(seq 0.05 0.01 1)
do
    for p_real in 0.5
    do
        p_plan_set=$( python3 -c "print($p_plan/2)")
        python3 ./Run.py -alg ATM_Robust            -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5  -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_RMDP              -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5  -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 0.5  -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure 1    -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        python3 ./Run.py -alg ATM_Control_Robust    -env_var $p_real -env_var_plan $p_plan_set -env_var_measure $p_plan_set -alpha_real 1 -alpha_plan 0.5 -alpha_measure -0.5 -env uMV -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
    done
done
wait

# ##########################################################
# #        Lucky-Unlucky Environment:
# ##########################################################


echo -e "\n\n============= C changing, uMV2  =============\n\n"
eps=1000
runs=1
i=0
folder_path="Data/Robust_Results/uMV2/"
get base environment
rsmall=0.8
alpha_real=0.0001
for cost in $(seq 0.3 0.01 0.8)
do
    python3 ./Run.py -alg ATM                   -m_cost $cost -env_var $rsmall -alpha_real $alpha_real -alpha_plan $alpha_real -env uMV2 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    i=$((i+1))
    if (("$i"=="$nmbr_cores"));
    then
        i=0
        wait
    fi 
done
wait

rsmall=0.8
p_real=$p_plan
alpha_real=0.0001
for cost in $(seq 0.3 0.01 0.8)
do
    python3 ./Run.py -alg ATM_Robust            -m_cost $cost -env_var $rsmall -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure $alpha_real   -env uMV2 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    python3 ./Run.py -alg ATM_RMDP              -m_cost $cost -env_var $rsmall -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure $alpha_real   -env uMV2 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    python3 ./Run.py -alg ATM_Control_Robust    -m_cost $cost -env_var $rsmall -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure $alpha_real   -env uMV2 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    python3 ./Run.py -alg ATM_Control_Robust    -m_cost $cost -env_var $rsmall -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure 1             -env uMV2  -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    python3 ./Run.py -alg ATM_Control_Robust    -m_cost $cost -env_var $rsmall -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure -$alpha_real  -env uMV2 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
    wait
done
wait

# ##########################################################
# #        Drone Environment:
# ##########################################################


echo -e "\n\n============= Alpha_real & Alpha_plan different, Drone =============\n\n"
folder_path="Data/Robust_Results/Drone/"
eps=100
runs=1
i=0
Get certain env:
python3 ./Run.py -alg ATM -env Drone -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path
for alpha_plan in 0.5 0.75 1.0
do
    alpha_plan=1
    for alpha_real in 0.3 0.35 0.7 0.85 0.95
    do
        echo -e "Alpha_plan = $alpha_plan , Alpha_real = $alpha_real "
        python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Drone -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        i=$((i+1))
        if (("$i"=="$nmbr_cores"));
        then
            i=0
            wait
        fi 
    done
done
wait
alpha_plan=0.5
for mcost in 0.1
do
    for alpha_real in $(seq 1  0.05 1.001)
    do
        echo -e "Alpha_plan = $alpha_plan , Alpha_real = $alpha_real, cost = $mcost "
        python3 ./Run.py -alg ATM                        -alpha_real $alpha_real -alpha_plan $alpha_plan                                -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        python3 ./Run.py -alg ATM_RMDP                   -alpha_real $alpha_real -alpha_plan $alpha_plan                                -env Drone -m_cost 0.1 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        python3 ./Run.py -alg ATM_Robust                 -alpha_real $alpha_real -alpha_plan $alpha_plan                                -env Drone -m_cost 0.1 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        ### Pessimistic
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure $alpha_plan     -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        ### Average
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 1               -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        ### Optimistic
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure -$alpha_plan    -env Drone -m_cost 0.1 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        # wait
    done
    wait
done
wait

echo -e "\n\n============= Alpha_real & Alpha_plan same, Drone =============\n\n"
folder_path="Data/Robust_Results/Drone/"
eps=100
runs=1
i=0
Get certain env:
python3 ./Run.py -alg ATM -env Drone -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path
for alpha_plan in 0.5 0.75 1.0
do
    alpha_plan=1
    for alpha_real in 0.3 0.35 0.7 0.85 0.95
    do
        echo -e "Alpha_plan = $alpha_plan , Alpha_real = $alpha_real "
        python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Drone -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        i=$((i+1))
        if (("$i"=="$nmbr_cores"));
        then
            i=0
            wait
        fi 
    done
done
wait
for mcost in 0.05
do
    for alpha_real in $(seq 0.6 0.05 1.001)
    do
        echo -e "Alpha_plan = $alpha_real , Alpha_real = $alpha_real, cost = $mcost "
        python3 ./Run.py -alg ATM                        -alpha_real $alpha_real -alpha_plan $alpha_real                                -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        # wait
        python3 ./Run.py -alg ATM_RMDP                   -alpha_real $alpha_real -alpha_plan $alpha_real                                -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        python3 ./Run.py -alg ATM_Robust                 -alpha_real $alpha_real -alpha_plan $alpha_real                                -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        ### Pessimistic
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure $alpha_real     -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        ### Average
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure 1               -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
        ### Optimistic
        python3 ./Run.py -alg ATM_Control_Robust         -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure -$alpha_real    -env Drone -m_cost $mcost -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
        wait
    done
    wait
done
wait


##########################################################
#        Additional (unused) experiments:
##########################################################



# echo -e "\n\n============= Alpha_real changing, Maintenace  =============\n\n"
# folder_path="Data/Robust_Results/Maintenance/"
# eps=250
# runs=1
# i=0
# # for alpha_plan in $(seq 0.5 0.1 0.8)
# # do
# #     for alpha_real in $(seq 0.5 0.02 1.001)
# #     do
# #         python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Maintenance -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
# #         i=$i+1
# #         if (($i==6))
# #         then
# #             i=0
# #             wait
# #         fi 
# #     done
# # done

# for alpha_plan in $(seq 0.5 0.1 0.8)
# do
#     for alpha_real in $(seq 0.5 0.02 1.001)
#     do
#         python3 ./Run.py -alg ATM_RMDP              -alpha_real $alpha_real -alpha_plan $alpha_plan                             -env Maintenance -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
#         python3 ./Run.py -alg ATM_Robust            -alpha_real $alpha_real -alpha_plan $alpha_plan                             -env Maintenance -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure $alpha_plan  -env Maintenance -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &      
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 1            -env Maintenance -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure -$alpha_plan -env Maintenance -nmbr_eps $eps -nmbr_runs $runs -rep $folder_path &
#         # Note: alpha_measure = alpha_plan!
#         wait
#     done
# done

# Experiments on small frozen lake, for different variants:
# echo -e "\n\n============= Alpha_real & Alpha_plan different, 12x12 Frozen Lake =============\n\n"
# folder_path="Data/Temp1/Lake/"
# eps=250
# runs=1
# i=0
# for alpha_plan in $(seq 0.65 0.05 0.8001)
# do
#     for alpha_real in $(seq 0.65 0.01 1.001)
#     do
#         python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_plan                     -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#         i=$i+1
#         if (($i==6))
#         then
#             i=0
#             wait
#         fi 
#     done
# done

# for alpha_plan in 0.65
# do
#     for alpha_real in $(seq 0.65 0.01 1.001)
#     do
#         python3 ./Run.py -alg ATM_RMDP              -alpha_real $alpha_real -alpha_plan $alpha_plan                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#         python3 ./Run.py -alg ATM_Robust            -alpha_real $alpha_real -alpha_plan $alpha_plan                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure $alpha_plan  -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure 1            -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#         python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_plan -alpha_measure -$alpha_plan -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#         wait
#     done
# done
# wait

# echo -e "\n\n============= Alpha_real and Alpha_plan equal, 12x12 Frozen Lake =============\n\n"
# folder_path="Data/Temp1/Lake/"
# eps=1000
# runs=1
# i=0
# for alpha_real in $(seq 0.65 0.01 1.001)
# do
#     python3 ./Run.py -alg ATM                   -alpha_real $alpha_real -alpha_plan $alpha_real                     -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     i=$i+1
#     if (($i==6))
#     then
#         i=0
#         wait
#     fi 
# done


# for alpha_real in $(seq 0.65 0.01 1.001)
# do
#     python3 ./Run.py -alg ATM_RMDP              -alpha_real $alpha_real -alpha_plan $alpha_real                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_Robust            -alpha_real $alpha_real -alpha_plan $alpha_real                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure $alpha_real  -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure 1            -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_Control_Robust    -alpha_real $alpha_real -alpha_plan $alpha_real -alpha_measure -$alpha_real -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     wait
# done

# wait

# echo -e "\n\n============= Beta changing, 12x12 Frozen Lake =============\n\n"
# folder_path="Data/Temp1/Lake/"
# eps=1000
# runs=1
# i=0
# alpha=0.8
# # make env
# python3 ./Run.py -alg ATM -alpha_real $alpha -alpha_plan $alpha -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path
# for beta in $(seq 0 0.05 0.5)
# do  
#     python3 ./Run.py -alg ATM                -beta $beta -alpha_real $alpha -alpha_plan $alpha                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_RMDP           -beta $beta -alpha_real $alpha -alpha_plan $alpha                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_Robust         -beta $beta -alpha_real $alpha -alpha_plan $alpha                             -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_Control_Robust -beta $beta -alpha_real $alpha -alpha_plan $alpha -alpha_measure $alpha       -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_Control_Robust -beta $beta -alpha_real $alpha -alpha_plan $alpha -alpha_measure 1            -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     python3 ./Run.py -alg ATM_Control_Robust -beta $beta -alpha_real $alpha -alpha_plan $alpha -alpha_measure -$alpha      -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 12 -rep $folder_path &
#     wait
#     # Note: alpha_measure = alpha_plan!
# done
# wait

echo -e "\n\n============= RUNS COMPLETED =============\n\n"


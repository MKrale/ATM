# ATM Repository

Repository containing code, as well as gathered data, as used for the paper

> *Anonymous*  
> Robust Active Measuring under Model Uncertainty  
> Submitted to AAAI 2024


## Contents

This repository contains the following files:

Code:

  - **ACNO_Planning.py**      : Code containing all planning algorithms used in the paper;
  - **Plot_Data.ipynb**       : Code for plotting data (with a **matplotlibrc** file to set formatting);
  - **Run.py**                : Code for automatically running agents on environments & recording their data;
  - **RunAll.sh**             : Bash file for automatically running all experiments in the paper;
  - **Requirements.text**     : File with required python dependencies;

Folders:

  - **AM_Gyms**             : Contains all code related to setting up and learning models, as used by the planning algorithms.
  - **Data**                : Contains gathered data, including analysed data & plots.
  - **Baselines**           : Contains code for all baseline algorithms used while testing.

## Getting started

After cloning this repository:

1. create a virtualenv and activate it
```bash
cd ATM/
python3 -m venv .venv
source .venv/bin/activate
```
2. install the dependencies
```bash
pip install -r requirements.txt
```

## How to run

All algorithms can be run using the Run.py file from command line. Running 'python Run.py -h' gives an overview of the functionaliality.

As an example, starting a run looks something like:

```bash
python Run.py -alg ATM_Control_Robust -env Drone -alpha_plan 0.5 -alpha_real 1 -alpha_measure 0.8 -nmbr_eps 100
```

This command runs the CR-ATM algorithm on the Drone environment with $\alpha = 1, \alpha_p = 0.5$, and $\mathcal{M}_\text{CR}$ with dynamics parametrized an RMDP with $\alpha=0.8$.
Thus, CR-ATM-avg uses alpha_measure 1, CR-ATM-pes uses alpha_measure = alpha_plan, and CR-ATM-opt uses alhpa_measure = - alpha_plan (hard-coded).
To run all experiments from the paper at once, run the following:

```bash
bash ./Runall.sh
```

Note that due to size constraints, this repository does not contain pre-computed drone environments, which means running this file might take a long time.
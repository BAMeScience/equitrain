#! /bin/bash

accelerate launch --config_file mace-alex-mptraj-multigpu.yaml mace-alex-mptraj.py

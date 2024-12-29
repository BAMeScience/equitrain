#! /bin/bash

NODE_RANK=1

accelerate launch --config_file mace-alex-mptraj-multinode.yaml --machine_rank $NODE_RANK mace-alex-mptraj.py

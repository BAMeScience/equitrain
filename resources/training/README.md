# Equitrain MACE Distributed Training Scripts

This directory contains scripts and configuration files for running distributed training of the `Equitrain` package with `MACE` on Alexandria and MPTraj data sets. These scripts support both single-node multi-GPU and multi-node multi-GPU setups using the `Accelerate` library. Additionally, the `mace-alex-mptraj-multinode.sh` script includes a SLURM-compatible version for HPC clusters.

---

## File Descriptions

### **1. Training Scripts**
#### `mace-alex-mptraj-multigpu.sh`
- Bash script to launch distributed training on a single machine with multiple GPUs.
- Uses the `mace-alex-mptraj-multigpu.yaml` configuration file.

#### `mace-alex-mptraj-multinode.sh`
- Bash script to launch distributed training across multiple nodes.
- Requires `mace-alex-mptraj-multinode.yaml` for configuration.
- Includes `NODE_RANK` as an environment variable to specify the rank of the node.

#### `mace-alex-mptraj-multinode-slurm.sh`
- SLURM-compatible version of the multi-node script.
- Automatically sets up environment variables for distributed training in SLURM-managed HPC clusters.
- Configurable via SLURM directives (e.g., `--nodes`, `--gres=gpu`).

---

### **2. Configuration Files**
#### `mace-alex-mptraj-multigpu.yaml`
- Configuration file for single-node, multi-GPU training.
- Key options:
  - `gpu_ids`: Specifies GPUs to use (e.g., `0,1,2,3`).
  - `num_processes`: Number of processes (matches number of GPUs).
  - `distributed_type`: Set to `MULTI_GPU`.

#### `mace-alex-mptraj-multinode.yaml`
- Configuration file for multi-node, multi-GPU training.
- Key options:
  - `machine_rank`: Specifies the rank of the machine in the distributed setup.
  - `main_process_ip`: IP address of the primary node.
  - `num_machines`: Number of nodes in the distributed setup.
  - `same_network`: Ensures nodes are on the same network for communication.

---

## Usage Instructions

### **1. Single-Node Multi-GPU Training**
1. Edit `mace-alex-mptraj-multigpu.yaml` to match your hardware setup (e.g., GPU IDs, number of processes).
2. Run the script:
   ```bash
   bash mace-alex-mptraj-multigpu.sh

### **2. Multi-Node Multi-GPU Training**
1. Configure `mace-alex-mptraj-multinode.yaml`:
   - Set `main_process_ip` to the IP address of the primary node.
   - Update `num_machines` and `machine_rank` as needed.
2. Run on each node, specifying `NODE_RANK` (e.g., `0` for primary node):
   ```bash
   bash mace-alex-mptraj-multinode.sh
Copy code


### **3. SLURM Multi-Node Training**
1. Adjust SLURM directives in `mace-alex-mptraj-multinode-slurm.sh` as needed (e.g., `--nodes`, `--gres=gpu`).
2. Submit the script to the SLURM scheduler:
   ```bash
   sbatch mace-alex-mptraj-multinode-slurm.sh
Copy code


## Troubleshooting
- **Connectivity Issues (Multi-Node):**
  - Ensure all nodes are on the same network and can communicate over the specified `main_process_port`.
- **GPU Utilization:**
  - Use `nvidia-smi` to verify that GPUs are being used as expected.
- **Debugging:**
  - Enable `debug: true` in the YAML configuration for detailed logs.

---

## References
- [Hugging Face Accelerate Documentation](https://huggingface.co/docs/accelerate/index)
- [SLURM Workload Manager](https://slurm.schedmd.com/documentation.html)


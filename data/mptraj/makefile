
N_JOBS = 20

# ------------------------------------------------------------------------------

all: mptraj

# ------------------------------------------------------------------------------

MPtrj_2022.9_full.json:
	wget -O MPtrj_2022.9_full.json "https://figshare.com/ndownloader/files/41619375"

mptraj_train.xyz: MPtrj_2022.9_full.json
	python 01_json_to_xyz.py

mptraj: mptraj_train.xyz
	python 02_xyz_to_hdf5.py

# ------------------------------------------------------------------------------

.DELETE_ON_ERROR:

.SECONDARY:

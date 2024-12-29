
N_JOBS = 20

# ------------------------------------------------------------------------------

all: alexandria_mptraj

# Alexandria
# ------------------------------------------------------------------------------

3D:
	python 01_download.py

3D_out: 3D
	python 02_convert_to_xyz.py --src_dir 3D --dst_dir 3D_out --n-jobs $(N_JOBS)

# MPTraj
# ------------------------------------------------------------------------------

MPtrj_2022.9_full.json:
	wget -O MPtrj_2022.9_full.json "https://figshare.com/ndownloader/files/41619375"

3D_out.mptraj:
	python 02_mptraj_to_xyz.py --src_file MPtrj_2022.9_full.json --dst_dir 3D_out
	@touch $@

# Alexandria+MPTraj
# ------------------------------------------------------------------------------

alexandria_mptraj_train.xyz: 3D_out 3D_out.mptraj
	python 03_join.py --src_dir 3D_out --dst_prefix alexandria_mptraj --n_jobs $(N_JOBS)

alexandria_mptraj: alexandria_mptraj_train.xyz
	python 04_xyz_to_hdf5.py

# ------------------------------------------------------------------------------

.DELETE_ON_ERROR:

.SECONDARY:

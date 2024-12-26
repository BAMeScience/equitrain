
N_JOBS = 20

# ------------------------------------------------------------------------------

all: alexandria

# ------------------------------------------------------------------------------

3D:
	python 01_download.py

3D_out: 3D
	python 02_convert_to_xyz.py --src_dir 3D --dst_dir 3D_out --n-jobs $(N_JOBS)

alexandria_train.xyz: 3D_out
	python 03_join.py --src_dir 3D_out --dst_prefix alexandria --n_jobs $(N_JOBS)

alexandria: alexandria_train.xyz
	python 04_xyz_to_hdf5.py

# ------------------------------------------------------------------------------

.DELETE_ON_ERROR:

.SECONDARY:

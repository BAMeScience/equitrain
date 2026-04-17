TRAIN_SUBSETS ?= --all
VAL_SUBSETS ?= --all
SALEX_SUBSETS ?= train val
OUTPUT_DIR ?= omat24

.PHONY: all train val salex extract extract-train extract-val extract-salex clean

# ------------------------------------------------------------------------------

all: $(OUTPUT_DIR)/.stats-complete

# ------------------------------------------------------------------------------

train:
	bash 00_download_omat24.sh --split train $(TRAIN_SUBSETS)

val:
	bash 00_download_omat24.sh --split val $(VAL_SUBSETS)

salex:
	bash 00_download_omat24.sh --split salex $(SALEX_SUBSETS)

# ------------------------------------------------------------------------------

extract: extract-train extract-val

extract-train: train
	@mkdir -p train
	@for archive in train/*.tar.gz; do \
		[ -e "$$archive" ] || continue; \
		echo "extracting $$archive"; \
		tar -xzf "$$archive" -C train; \
	done

extract-val: val
	@mkdir -p val
	@for archive in val/*.tar.gz; do \
		[ -e "$$archive" ] || continue; \
		echo "extracting $$archive"; \
		tar -xzf "$$archive" -C val; \
	done

extract-salex: salex
	@mkdir -p salex
	@for archive in salex/*.tar.gz; do \
		[ -e "$$archive" ] || continue; \
		echo "extracting $$archive"; \
		tar -xzf "$$archive" -C salex; \
	done

# ------------------------------------------------------------------------------

$(OUTPUT_DIR)/train.h5: extract-train 01_aselmdb_to_hdf5.py
	python 01_aselmdb_to_hdf5.py --train-dir train --output-dir $(OUTPUT_DIR) --skip-valid

$(OUTPUT_DIR)/valid.h5: extract-val 01_aselmdb_to_hdf5.py
	python 01_aselmdb_to_hdf5.py --valid-dir val --output-dir $(OUTPUT_DIR) --skip-train

$(OUTPUT_DIR)/.stats-complete: $(OUTPUT_DIR)/train.h5 $(OUTPUT_DIR)/valid.h5 02_compute_statistics.py
	python 02_compute_statistics.py
	@touch $@

# ------------------------------------------------------------------------------

clean:
	rm -rf train val salex $(OUTPUT_DIR)

# ------------------------------------------------------------------------------

.DELETE_ON_ERROR:

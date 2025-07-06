

# ============================================================================
# GSPLAT QUALITY PROFILE PARAMETERS
# ----------------------------------------------------------------------------
# Each quality level is broken into its own block for maintainability
# ============================================================================

define gsplat-quality-test
    --iterations 3000 \
    --sh_degree 2
endef

define gsplat-quality-fast
    --iterations 5000 \
    --sh_degree 3
endef

define gsplat-quality-balanced
    --iterations 20000 \
    --position_lr_init 0.02 --position_lr_final 0.005 \
    --densify_from_iter 3000 --densify_until_iter 15000 \
    --densification_interval 500 --percent_dense 0.2 \
    --sh_degree 3
endef

define gsplat-quality-highdetail
    --iterations 50000 \
    --position_lr_init 0.01 --position_lr_final 0.001 \
    --densify_from_iter 5000 --densify_until_iter 45000 \
    --densification_interval 500 --percent_dense 0.3 \
    --lambda_dssim 0.2 \
    --sh_degree 4
endef

define gsplat-quality-ultrasharp
    --iterations 40000 \
    --position_lr_init 0.01 --position_lr_final 0.001 \
    --densify_from_iter 3000 --densify_until_iter 35000 \
    --densification_interval 300 --opacity_reset_interval 1200 \
    --densify_grad_threshold 0.0005 --percent_dense 0.25 \
    --lambda_dssim 0.3 --sh_degree 3
endef

define gsplat-quality-default
	--iterations 15000 \
	--save_iterations 15000 \
	--sh_degree 3
endef


define recipe-gsplat-transform
	echo "==================================================================="; \
	echo "GSPLAT TRANSFORM TARGET: $(@)"; \
	echo "Dependent: $(firstword $(^))"; \
	echo "==================================================================="; \
	TRANSFORM_STRING="$($(call ELEM5,$(@),2).gsplat.transform)" ; \
	if [ -n "$$TRANSFORM_STRING" ]; then \
		echo "Running transform with options: $$TRANSFORM_STRING"; \
		find $(@)/point_cloud -type f -name "point_cloud.ply" | while read ply_file; do \
			ply_dir=$$(dirname "$$ply_file"); \
			echo "Backing up $$ply_file -> $$ply_dir/point_cloud_before_transform.ply"; \
			cp "$$ply_file" "$$ply_dir/point_cloud_before_transform.ply"; \
			echo "Transforming $$ply_dir/point_cloud_before_transform.ply -> $$ply_dir/point_cloud.ply"; \
			poetry run python scripts/cli.py transform \
				-i "$$ply_dir/point_cloud_before_transform.ply" \
				-o "$$ply_dir/point_cloud.ply" \
				$$TRANSFORM_STRING; \
		done; \
	else \
		echo "No transform string defined. Skipping transform step."; \
	fi; \
	echo "==================================================================="; \
	echo "FINISHED TRANSFORM TARGET: $(@)"
endef


define recipe-gsplat-convert
	echo "Running GSPLAT CONVERT inside container..." ; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 gsplat \
	sh -c "find /$(@)/point_cloud -type f -name \"point_cloud.ply\" | while read ply_file; do \
	ply_dir=\$$(dirname \"\$$ply_file\"); \
	echo \"Converting to \$$ply_dir/point_cloud.splat\"; \
	python /opt/point-cloud-tools/convert.py \
		\"\$$ply_file\" \
		\"\$$ply_dir/point_cloud.splat\"; \
	done"
endef


define recipe-gsplat-get-sparse-and-transform
	echo "Gathering SPARSE from colmap" ; \
	rm -fr $(@)/sparse_from_colmap ; \
	mkdir -p $(@)/sparse_from_colmap ; \
	echo ">>> Converting colmap from BIN to TXT into $(@)"; \
	docker compose -f ./docker/docker-compose.yml run --rm --user 1000:1000 colmap \
	colmap model_converter \
		--input_path /$(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/colmap/$(call ELEM5,$(@),4)/sparse/ \
		--output_path /$(@)/sparse_from_colmap \
		--output_type TXT; \
	\
	echo ">>> Converting colmap from BIN to TXT into $(@)"; \
	docker compose -f ./docker/docker-compose.yml run --rm --user 1000:1000 colmap \
	colmap model_converter \
		--input_path /$(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/colmap/$(call ELEM5,$(@),4)/sparse/ \
		--output_path /$(@)/sparse_from_colmap/points3D.ply \
		--output_type PLY; \
	\
	echo ">>> Translating images and points" ; \
	rm -fr $(@)/after_transform; \
	mkdir -p $(@)/after_transform; \
	poetry run python scripts/cli.py transform-for-gsplat \
	-i $(@)/sparse_from_colmap \
	-o $(@)/after_transform \
	$($(call ELEM5,$(@),2).gsplat.transform) ; \
	\
	echo ">>> Converting centered/rotated TXT back to BIN in the same folder"; \
	docker compose -f ./docker/docker-compose.yml run --rm --user 1000:1000 colmap \
	colmap model_converter \
		--input_path /$(@)/after_transform \
		--output_path /$(@)/after_transform \
		--output_type BIN; \
	echo "Done with gathering and transforming" 
endef



# ============================================================================
# MAIN GSPLAT MODEL RECIPE
# ----------------------------------------------------------------------------
# Uses the quality type to pick the options block dynamically.
# ============================================================================

define recipe-gsplat-model
	@echo "==================================================================="; \
	echo "GSPLAT TARGET: $(@)"; \
	echo "Dependent: $(firstword $(^))"; \
	echo "Features: $(gsplat-model-$(call ELEM5,$(@),4))"; \
	echo "==================================================================="; \
	mkdir -p $(@); \
	{ \
	QUALITY_STR="$(gsplat-model-$(call ELEM5,$(@),4))"; \
	if echo "$$QUALITY_STR" | grep -q "fast"; then \
		GSPLAT_OPTS_RAW="$(gsplat-quality-fast)"; \
	elif echo "$$QUALITY_STR" | grep -q "balanced"; then \
		GSPLAT_OPTS_RAW="$(gsplat-quality-balanced)"; \
	elif echo "$$QUALITY_STR" | grep -q "highdetail"; then \
		GSPLAT_OPTS_RAW="$(gsplat-quality-highdetail)"; \
	elif echo "$$QUALITY_STR" | grep -q "test"; then \
		GSPLAT_OPTS_RAW="$(gsplat-quality-test)"; \
	elif echo "$$QUALITY_STR" | grep -q "ultrasharp"; then \
		GSPLAT_OPTS_RAW="$(gsplat-quality-ultrasharp)"; \
	else \
		GSPLAT_OPTS_RAW="$(gsplat-quality-default)"; \
	fi; \
	FLAT_GSPLAT_OPTS=$$(echo "$$GSPLAT_OPTS_RAW" | awk '{printf "%s ", $$0} END {print ""}'); \
	ITER_DIR=iteration_$$(echo "$$FLAT_GSPLAT_OPTS" | grep -oP '(?<=--iterations )\d+'); \
	echo "Running GSPLAT with options: $$FLAT_GSPLAT_OPTS"; \
	echo "Expecting final file: /$(@)/point_cloud/$$ITER_DIR/point_cloud.splat"; \
	\
	$(recipe-gsplat-get-sparse-and-transform) ; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 gsplat \
	python train.py \
		--source_path /$(@)/after_transform \
		--model_path /$(@) \
		--images /$(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/colmap/$(call ELEM5,$(@),4)/images \
		$$FLAT_GSPLAT_OPTS; \
	\
	$(recipe-gsplat-convert); \
	}; \
	echo "==================================================================="; \
	echo "FINISHED GSPLAT TARGET: $(@)"
endef

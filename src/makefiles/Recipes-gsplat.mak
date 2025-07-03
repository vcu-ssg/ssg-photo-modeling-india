

# ============================================================================
# GSPLAT QUALITY PROFILE PARAMETERS
# ----------------------------------------------------------------------------
# Each quality level is broken into its own block for maintainability
# ============================================================================

define gsplat-quality-fast
    --iterations 5000 \
    --position_lr_init 0.05 --position_lr_final 0.01 \
    --densify_from_iter 1000 --densify_until_iter 4000 \
    --densification_interval 200 --percent_dense 0.1 \
    --sh_degree 2
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
	elif echo "$$QUALITY_STR" | grep -q "ultrasharp"; then \
		GSPLAT_OPTS_RAW="$(gsplat-quality-ultrasharp)"; \
	else \
		GSPLAT_OPTS_RAW="$(gsplat-quality-default)"; \
	fi; \
	FLAT_GSPLAT_OPTS=$$(echo "$$GSPLAT_OPTS_RAW" | awk '{printf "%s ", $$0} END {print ""}'); \
	ITER_DIR=iteration_$$(echo "$$FLAT_GSPLAT_OPTS" | grep -oP '(?<=--iterations )\d+'); \
	echo "Running GSPLAT with options: $$FLAT_GSPLAT_OPTS"; \
	echo "Expecting final file: /$(@)/point_cloud/$$ITER_DIR/point_cloud.splat"; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 gsplat \
	python train.py \
		--source_path /$(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/colmap/$(call ELEM5,$(@),4)/sparse \
		--model_path /$(@) \
		--images /$(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/colmap/$(call ELEM5,$(@),4)/images \
		$$FLAT_GSPLAT_OPTS; \
	echo "Running convert loop inside single container..."; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 gsplat \
	sh -c "find /$(@)/point_cloud -type f -name \"point_cloud.ply\" | while read ply_file; do \
		ply_dir=\$$(dirname \"\$$ply_file\"); \
		echo \"Converting \$$ply_file to \$$ply_dir/point_cloud.splat\"; \
		python /opt/point-cloud-tools/convert.py \"\$$ply_file\" \"\$$ply_dir/point_cloud.splat\"; \
	done"; \
	}; \
	echo "==================================================================="; \
	echo "FINISHED GSPLAT TARGET: $(@)"
endef

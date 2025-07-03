
# ---------------------------
# Extractor options
# ---------------------------
define extractor-default
	--SiftExtraction.max_image_size 1600
endef

define extractor-highres
	--SiftExtraction.max_image_size 3200 \
	--SiftExtraction.max_num_features 65536 \
	--SiftExtraction.peak_threshold 0.002 \
	--SiftExtraction.edge_threshold 5 \
	--SiftExtraction.octave_resolution 4
endef

define extractor-adaptive
	--SiftExtraction.max_image_size 2400 \
	--SiftExtraction.domain_size_pooling 1
endef

# ---------------------------
# Matcher options
# ---------------------------
define matcher-default
	--SequentialMatching.overlap 10
endef

define matcher-fast
	--SequentialMatching.overlap 5
endef

define matcher-loop
	--SequentialMatching.overlap 10 \
	--SequentialMatching.loop_detection 1
endef

define matcher-guided
	--SequentialMatching.overlap 10 \
	--SiftMatching.guided_matching 1
endef

# ---------------------------
# Mapper options
# ---------------------------

# ---------------------------
# Mapper options
# ---------------------------

define mapper-fast
	--Mapper.min_num_matches 12 \
	--Mapper.abs_pose_min_num_inliers 30 \
	--Mapper.abs_pose_min_inlier_ratio 0.2 \
	--Mapper.init_min_tri_angle 4 \
	--Mapper.filter_max_reproj_error 6 \
	--Mapper.filter_min_tri_angle 1.5 \
	--Mapper.ba_local_max_num_iterations 15
endef

define mapper-default
	--Mapper.min_num_matches 15 \
	--Mapper.abs_pose_min_num_inliers 50 \
	--Mapper.abs_pose_min_inlier_ratio 0.25 \
	--Mapper.init_min_tri_angle 6 \
	--Mapper.filter_max_reproj_error 4 \
	--Mapper.filter_min_tri_angle 2 \
	--Mapper.ba_local_max_num_iterations 25
endef

define mapper-robust
	--Mapper.min_num_matches 20 \
	--Mapper.abs_pose_min_num_inliers 60 \
	--Mapper.abs_pose_min_inlier_ratio 0.3 \
	--Mapper.init_min_tri_angle 8 \
	--Mapper.filter_max_reproj_error 3 \
	--Mapper.filter_min_tri_angle 2.5 \
	--Mapper.ba_refine_principal_point 0 \
	--Mapper.ba_local_max_num_iterations 30
endef

define mapper-highquality
	--Mapper.min_num_matches 25 \
	--Mapper.abs_pose_min_num_inliers 80 \
	--Mapper.abs_pose_min_inlier_ratio 0.35 \
	--Mapper.init_min_tri_angle 10 \
	--Mapper.filter_max_reproj_error 2.5 \
	--Mapper.filter_min_tri_angle 3 \
	--Mapper.ba_global_max_num_iterations 60 \
	--Mapper.ba_local_max_num_iterations 40
endef


# ---------------------------
# Pipeline recipes
# ---------------------------
define recipe-colmap-prepare-images
	echo ">>> Preparing images & masks"; \
	mkdir -p $(@)/images $(@)/masks; \
	MASK_TYPE=$$(echo "$(1)" | sed -n 's/.*mask=\([^ ]*\).*/\1/p'); \
	if [ -z "$$MASK_TYPE" ]; then MASK_TYPE="open"; fi; \
	if [ "$$MASK_TYPE" != "open" ]; then \
		if echo "$(1)" | grep -q "mode=extractor"; then \
			echo ">>> [EXTRACTOR MODE] Generating masks with filter='$$MASK_TYPE'"; \
			poetry run python scripts/cli.py generate-masks \
			--images-dir=$(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/images \
			--output-mask-dir=$(@)/masks \
			--output-masked-image-dir= \
			--filter=$$MASK_TYPE \
			--workers=16 ; \
			cp $(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/images/*.* $(@)/images ; \
		elif echo "$(1)" | grep -q "mode=direct"; then \
			echo ">>> [DIRECT MODE] Generating masked images with filter='$$MASK_TYPE'"; \
			poetry run python scripts/cli.py generate-masks \
			--images-dir=$(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/images \
			--output-mask-dir=$(@)/masks \
			--output-masked-image-dir=$(@)/images \
			--filter=$$MASK_TYPE \
			--workers=16 ; \
		else \
			echo ">>> ERROR: mask=$$MASK_TYPE requires mode=extractor or mode=direct" >&2; \
			exit 1; \
		fi; \
	else \
		echo ">>> [OPEN MODE] No masking applied. Copying raw images."; \
		cp $(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/images/*.* $(@)/images ; \
	fi
endef

define recipe-colmap-feature-extracter
	echo ">>> Feature extractor with options: $(1) $(2)"; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 colmap \
	colmap feature_extractor \
		--database_path /$(@)/db.db \
		--image_path /$(@)/images \
		--ImageReader.single_camera 1 \
		--ImageReader.camera_model PINHOLE \
		--SiftExtraction.use_gpu 1 \
		--SiftExtraction.num_threads 16 \
		$(2) \
		$(1)
endef

define recipe-colmap-sequential-matcher
	echo "COLMAP sequential matcher with options: $(1)"; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 colmap \
	colmap sequential_matcher \
		--database_path /$(@)/db.db \
		--SiftMatching.use_gpu 1 \
		--SiftMatching.num_threads 16 \
		$(1)
endef

define recipe-colmap-cleaner-1
	echo ">>> Running cleaner strategy"
endef

define recipe-colmap-mapper
	echo "COLMAP mapper with options: $(1)"; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 colmap \
	colmap mapper \
		--database_path /$(@)/db.db \
		--image_path /$(call ELEM5,$(@),1)/$(call ELEM5,$(@),2)/images \
		--output_path /$(@) \
		--Mapper.num_threads 16 \
		$(1)
endef

define recipe-colmap-model-transformer
	echo ">>> Checking for transform string override..."; \
	TRANSFORM_STRING="$($(call ELEM5,$(@),2).transform)"; \
	if [ -n "$$TRANSFORM_STRING" ]; then \
		echo "Transform string for $(call ELEM5,$(@),2): $$TRANSFORM_STRING"; \
	else \
		echo "No transform string defined for $(call ELEM5,$(@),2). Aborting."; \
		exit 1; \
	fi; \
	echo ">>> Generating transform file at $(@)/0_aligned/transform.txt"; \
	mkdir -p $(@)/0_aligned; \
	poetry run python scripts/generate_transform.py $$TRANSFORM_STRING > $(@)/0_aligned/transform.txt; \
	echo ">>> Transform file contents:"; \
	cat $(@)/0_aligned/transform.txt; \
	echo ">>> Running colmap model_transformer with generated transform"; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 colmap \
	colmap model_transformer \
		--input_path /$(@)/0 \
		--output_path /$(@)/0_aligned \
		--transform_path /$(@)/0_aligned/transform.txt; \
	echo ">>> Backing up original model to $(@)/0_before_transform"; \
	rm -rf $(@)/0_before_transform; \
	mv $(@)/0 $(@)/0_before_transform; \
	echo ">>> Moving aligned model to $(@)/0"; \
	mv $(@)/0_aligned $(@)/0
endef


define recipe-colmap-model-converter
	echo "COLMAP model converter" ; \
	docker compose -f ./docker/docker-compose.yml \
	run --rm --user 1000:1000 colmap \
	colmap model_converter \
		--input_path /$(@)/0 \
		--output_path /$(@)/0/points3D.ply \
		--output_type PLY ; \
	echo "Done. COLMAP model converter"
endef

define recipe-add-normals-to-ply
	echo "Adding normals to PLY to: $(@)/0/points3D.ply ..."; \
	poetry run \
	python -c 'import pymeshlab; \
ms = pymeshlab.MeshSet(); \
ms.load_new_mesh("$(@)/0/points3D.ply"); \
ms.compute_normal_for_point_clouds(); \
ms.save_current_mesh("$(@)/0/points3D_with_normals.ply", \
binary=True, \
save_vertex_normal=True, \
save_vertex_color=True);'
endef


define recipe-colmap-complete-folders
	echo "-------------------------------------------------------------------"; \
	echo "Complete folder setup with links and copies"; \
	echo "Target: $(@)"; \
	echo "Depend: $(firstword $(^))"; \
	echo "-------------------------------------------------------------------"; \
	cp $(@)/0/points3D_with_normals.ply $(@)/0/point_cloud.ply; \
	cp -r $(@)/0 $(@)/sparse; \
	cp $(@)/0/points3D_with_normals.ply $(@)/sparse/points3D.ply; \
	if [ ! -e "$(@)/images" ]; then \
	  ln -s ../../images $(@)/images; \
	  echo "Created symlink $(@)/images -> ../../images"; \
	else \
	  echo "Link or folder $(@)/images already exists"; \
	fi
endef

# ---------------------------
# Dynamic master recipe
# ---------------------------
define recipe-colmap-model
	@echo "==================================================================="; \
	echo "COLMAP MODEL: $(call ELEM5,$(@),4)"; \
	echo "Target: $(@)"; \
	echo "Depend: $(firstword $(^))"; \
	echo "==================================================================="; \
	{ \
	QUALITY_STR="$(colmap-model-$(call ELEM5,$(@),4))"; \
	\
	if echo "$$QUALITY_STR" | grep -q "extract=highres"; then EXTRACT_OPTS="$(extractor-highres)"; \
	elif echo "$$QUALITY_STR" | grep -q "extract=adaptive"; then EXTRACT_OPTS="$(extractor-adaptive)"; \
	else EXTRACT_OPTS="$(extractor-default)"; fi; \
	\
	if echo "$$QUALITY_STR" | grep -q "match=loop"; then MATCH_OPTS="$(matcher-loop)"; \
	elif echo "$$QUALITY_STR" | grep -q "match=guided"; then MATCH_OPTS="$(matcher-guided)"; \
	elif echo "$$QUALITY_STR" | grep -q "match=fast"; then MATCH_OPTS="$(matcher-fast)"; \
	else MATCH_OPTS="$(matcher-default)"; fi; \
	\
	if echo "$$QUALITY_STR" | grep -q "mapper=robust"; then MAPPER_OPTS="$(mapper-robust)"; \
	elif echo "$$QUALITY_STR" | grep -q "mapper=highquality"; then MAPPER_OPTS="$(mapper-highquality)"; \
	elif echo "$$QUALITY_STR" | grep -q "mapper=fast"; then MAPPER_OPTS="$(mapper-fast)"; \
	else MAPPER_OPTS="$(mapper-default)"; fi; \
	\
	if echo "$$QUALITY_STR" | grep -q "mask=" && echo "$$QUALITY_STR" | grep -q "mode=extractor"; then \
		MASK_PATH_OPT="--ImageReader.mask_path /$(@)/masks"; \
	else \
		MASK_PATH_OPT=""; \
	fi; \
	\
	echo "Extractor opts: $$EXTRACT_OPTS"; \
	echo "Matcher opts: $$MATCH_OPTS"; \
	echo "Mapper opts: $$MAPPER_OPTS"; \
	echo "Mask path: $$MASK_PATH_OPT"; \
	\
	$(call recipe-colmap-prepare-images,$(colmap-model-$(call ELEM5,$(@),4))); \
	\
	$(call recipe-colmap-feature-extracter,$$EXTRACT_OPTS,$$MASK_PATH_OPT); \
	$(call recipe-colmap-sequential-matcher,$$MATCH_OPTS); \
	if echo "$$QUALITY_STR" | grep -q "filter=clean"; then \
		$(recipe-colmap-cleaner-1); \
	fi; \
	$(call recipe-colmap-mapper,$$MAPPER_OPTS); \
	$(recipe-colmap-model-transformer); \
	$(recipe-colmap-model-converter); \
	$(recipe-add-normals-to-ply); \
	$(recipe-colmap-complete-folders); \
	}; \
	echo "==================================================================="; \
	echo "FINISHED COLMAP MODEL: $(call ELEM5,$(@),4)"; \
	echo "Target: $(@)"; \
	echo "Depend: $(firstword $(^))"
endef


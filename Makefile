

include src/makefiles/Sources-sync.mak
include src/makefiles/Recipes-utils.mak
include src/makefiles/Recipes-colmap.mak
include src/makefiles/Recipes-gsplat.mak
include src/makefiles/Recipes.mak


project-roots := hanuman

hanuman.sources := Dec13at10-33PM-HanumanClose-poly-20250703T115334Z-1-001.zip hunaman.md
hanuman.transform := X -45 Y 90 Z 90

model-roots := 0
colmap-model-0 := mask=open extract=default match=fast mapper=fast
gsplat-model-0 := quality=default


test: projects/hanuman/colmap/0/rotate
.PHONY: projects/hanuman/colmap/0/rotate
projects/hanuman/colmap/0/rotate : ; $(recipe-colmap-model-aligner)


$(foreach project,$(project-roots),$(eval projects/$(project)/sources : ; $$(recipe-download-project-sources)))
$(foreach project,$(project-roots),$(eval projects/$(project)/images : projects/$(project)/sources ; $$(recipe-unzip-project-images)))
$(foreach project,$(project-roots),$(foreach model,$(model-roots),$(eval projects/$(project)/colmap/$(model) : projects/$(project)/images ; $$(recipe-colmap-model))))
$(foreach project,$(project-roots),$(foreach model,$(model-roots),$(eval projects/$(project)/gsplat/$(model) : projects/$(project)/colmap/$(model) ; $$(recipe-gsplat-model))))

realclean:
	rm -fr projects/

## Interactive shell targets for debugging

container-roots := colmap gsplat openmvg openmvs

# rebuild containers - clean old image prune layers
$(foreach container,$(container-roots),rebuild-$(container)) :
	docker builder prune --all --force
	docker rmi -f $(call ELEM1,$(@),2) || true
	make build-$(call ELEM1,$(@),2)

# build containers
$(foreach container,$(container-roots),build-$(container)) :
	 COMPOSE_BAKE=true docker compose -f ./docker/docker-compose.yml build $(call ELEM1,$(@),2)

# open shell inside container
$(foreach container,$(container-roots),shell-$(container)) :
	docker compose -f ./docker/docker-compose.yml run --rm $(call ELEM1,$(@),2) bash



build-reports:
	rm -f reports/prj/*.*
	poetry run python scripts/cli.py \
	generate-project-reports \
	--projects-root=projects \
	--report-qmds=reports/prj \
	--report-data=docs/data
	cd reports && poetry run quarto render

preview : build-reports
	cd reports && poetry run quarto preview



include src/makefiles/Sources-sync.mak
include src/makefiles/Recipes-utils.mak
include src/makefiles/Recipes-containers.mak
include src/makefiles/Recipes-colmap.mak
include src/makefiles/Recipes-gsplat.mak
include src/makefiles/Recipes.mak


project-roots := hanuman

hanuman.sources := Dec13at10-33PM-HanumanClose-poly-20250703T115334Z-1-001.zip hunaman.md
hanuman.colmap.transform := TX 0 TY 0 TZ 0 RX 0 RY 0 RZ 0 S 1
hanuman.gsplat.transform := --rx 90 --ry 90 --rz -120 --tx 0

model-roots := 0
colmap-model-0 := mask=open extract=default match=default mapper=default filter=default
gsplat-model-0 := quality=default


$(foreach project,$(project-roots),$(eval projects/$(project)/sources : ; $$(recipe-download-project-sources)))
$(foreach project,$(project-roots),$(eval projects/$(project)/images : projects/$(project)/sources ; $$(recipe-unzip-project-images)))
$(foreach project,$(project-roots),$(foreach model,$(model-roots),$(eval projects/$(project)/colmap/$(model) : projects/$(project)/images ; $$(recipe-colmap-model))))
$(foreach project,$(project-roots),$(foreach model,$(model-roots),$(eval projects/$(project)/gsplat/$(model) : projects/$(project)/colmap/$(model) ; $$(recipe-gsplat-model))))

realclean:
	rm -fr projects/


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





SOURCES_LOCAL_DIR = ./sources
SOURCES_REMOTE_TAG = sources

help:
	@echo help goes here!

$(SOURCES_REMOTE_TAG)-create:
	gh release create $(SOURCES_REMOTE_TAG) -t "Source zip files" -n "Large asset storage for images and videos"

$(SOURCES_REMOTE_TAG)-list:
	@echo "Available assets on release '$(SOURCES_REMOTE_TAG)':"
	@GH_PAGER= gh release view $(SOURCES_REMOTE_TAG) --json assets --jq '.assets[].name'
 
$(SOURCES_REMOTE_TAG)-fetch:
	@mkdir -p $(SOURCES_LOCAL_DIR)
	@echo "Checking and downloading missing assets from release '$(SOURCES_REMOTE_TAG)'..."
	@for asset in $(shell GH_PAGER= gh release view $(SOURCES_REMOTE_TAG) --json assets --jq '.assets[].name'); do \
		if [ -f "$(SOURCES_LOCAL_DIR)/$$asset" ]; then \
			echo "✓ $$asset already exists."; \
		else \
			echo "↓ Downloading $$asset..."; \
			curl -L -o "$(SOURCES_LOCAL_DIR)/$$asset" \
				"https://github.com/$(shell gh repo view --json nameWithOwner --jq .nameWithOwner)/releases/download/$(SOURCES_REMOTE_TAG)/$$asset"; \
		fi \
	done

$(SOURCES_REMOTE_TAG)-upload:
	@echo "Uploading local files in '$(SOURCES_LOCAL_DIR)/' not yet in release '$(SOURCES_REMOTE_TAG)'..."
	@existing_assets=$$(GH_PAGER= gh release view $(SOURCES_REMOTE_TAG) --json assets --jq '.assets[].name'); \
	for file in $(SOURCES_LOCAL_DIR)/*; do \
		filename=$$(basename $$file); \
		if echo "$$existing_assets" | grep -qx "$$filename"; then \
			echo "✓ $$filename already uploaded to release."; \
		else \
			echo "↑ Uploading $$filename to GitHub release..."; \
			gh release upload $(SOURCES_REMOTE_TAG) "$$file"; \
		fi \
	done

$(SOURCES_REMOTE_TAG)-sync: sources-fetch sources-upload
	@echo "✅ Synchronization complete: local '$(SOURCES_LOCAL_DIR)' folder and GitHub release '$(SOURCES_REMOTE_TAG)' are now in sync."


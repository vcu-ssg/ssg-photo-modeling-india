
define recipe-download-project-sources
	echo ">>> downloading sources: $(^)"
	mkdir -p $(@)
	echo "Checking and downloading missing assets..."
	for asset in $(hanuman.sources); do \
		if [ -f "$(@)/$$asset" ]; then \
			echo "✓ $$asset already exists."; \
		else \
			echo "↓ Downloading $$asset..."; \
			curl -L -o "$(@)/$$asset" \
				"https://github.com/$(shell gh repo view --json nameWithOwner --jq .nameWithOwner)/releases/download/sources/$$asset"; \
		fi \
	done
endef



define recipe-unzip-project-images
	@echo "Creating directory $(@)..."
	@mkdir -p "$(@)"
	@echo "Unzipping all zip files from $(@)/../sources into $(@)..."
	@for zipfile in "$(@)/../sources"/*.zip; do \
		if [ -f "$$zipfile" ]; then \
			echo "→ Unzipping $$zipfile into $(@)..."; \
			unzip -j -o "$$zipfile" -d "$(@)"; \
		else \
			echo "⚠ No zip files found in $(@)/../sources"; \
		fi \
	done
endef


# TODO: make sure this runs
MAX_LINE_SIZE := 120

train: 
	python training.py

test:
	@eval "$$(pyenv init -)" && \
	pyenv activate $(VENV_NAME)

	echo $$(pyenv which python)
	$$(pyenv which python) Scripts/DataPrep/DataPrep.py

venv: 
	@eval "$$(pyenv init -)" && \
	pyenv activate $(VENV_NAME)

#	echo $$(pyenv which python)


# ======================= Initial setup =======================


download-data: 
	mkdir -p Data
# Download face data
	wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar -P Data/
	tar -xf Data/imdb_crop.tar -C Data/
	rm Data/imdb_crop.tar

# Download matlab metadata
	wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_meta.tar -P Data/
	tar -xf Data/imdb_meta.tar -C Data/
	rm Data/imdb_meta.tar

# TODO: rmake sure this runs
generate-metadata-and-cropped-images: venv
	poetry run python generate_metadata_and_cropped_images.py


make pre-lint:
	poetry run black --preview --line-length $(MAX_LINE_SIZE) .
	poetry run isort --profile=black --line-length $(MAX_LINE_SIZE) .
	poetry run flake8 --max-line-length $(MAX_LINE_SIZE) .

make lint:
	poetry run black --preview --line-length $(MAX_LINE_SIZE) --check .
	poetry run isort --profile=black --line-length $(MAX_LINE_SIZE) --check-only .
	poetry run flake8 --max-line-length=$(MAX_LINE_SIZE) 
MAX_LINE_SIZE := 120

train: 
	python training.py

venv: 
	@eval "$$(pyenv init -)" && \
	pyenv activate $(VENV_NAME)

#	echo $$(pyenv which python)


# ======================= Initial setup =======================


make generate-metadata:
	poetry run python scripts/generate_metadata.py

make pre-lint:
	poetry run black --preview --line-length $(MAX_LINE_SIZE) .
	poetry run isort --profile=black --line-length $(MAX_LINE_SIZE) **/*.py
	poetry run flake8 --max-line-length $(MAX_LINE_SIZE) **/*.py

make lint:
	poetry run black --preview --line-length $(MAX_LINE_SIZE) --check .
	poetry run isort --profile=black --line-length $(MAX_LINE_SIZE) --check-only **/*.py
	poetry run flake8 --max-line-length=$(MAX_LINE_SIZE) **/*.py

make tensorboard:
	poetry run tensorboard --logdir=tb_logs
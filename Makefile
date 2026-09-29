test:
	uv sync --group dev
	$(PYTHON) -m unittest discover tests

export-silero-onnx:
	PYTHONPATH=. uv run scripts/export_silero_onnx.py

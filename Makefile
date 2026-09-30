test:
	uv sync --group dev
	uv run -m unittest discover tests

export-silero-onnx:
	PYTHONPATH=. uv run scripts/export_silero_onnx.py

benchmark-dict-lookups:
	uv run scripts/benchmark_dict_lookups.py --limit 5000 --repeat 5

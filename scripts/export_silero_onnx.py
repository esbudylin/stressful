# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "silero-stress",
#     "onnx==1.21.0",
#     "onnxruntime==1.23.2",
#     "numpy<1.24",
#     "torch>=2.10,<2.11",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
# ///

"""Export silero-stress to ONNX Runtime + auxiliary data.

Writes models and data consumed by stressful.silero.SileroAccentor.
"""

import json
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from silero_stress import load_accentor

from stressful.settings import SILERO_MODEL_DIR

HOMO_DUMMY_SEQ = 16
DYNAMIC = {
    "input_ids": {0: "batch", 1: "seq"},
    "homo_start": {0: "batch"},
    "homo_end": {0: "batch"},
    "logits": {0: "batch"},
}

CLF_DYNAMIC = {"emb": {0: "batch"}, "logits": {0: "batch"}}


def export_onnx(model, args, path, input_names, output_names, dynamic_axes):
    logging.info("Exporting %s", path)
    torch.onnx.export(
        model,
        args,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        opset_version=17,
    )


def export_models(accentor):
    model = accentor.homosolver.model

    input_ids = torch.zeros(1, HOMO_DUMMY_SEQ, dtype=torch.long)
    homo_start = torch.tensor([1], dtype=torch.long)
    homo_end = torch.tensor([HOMO_DUMMY_SEQ - 2], dtype=torch.long)

    homo_path = os.path.join(SILERO_MODEL_DIR, "homo.onnx")
    quantized_path = os.path.join(SILERO_MODEL_DIR, "homo_int8.onnx")

    export_onnx(
        model,
        (input_ids, homo_start, homo_end),
        homo_path,
        ["input_ids", "homo_start", "homo_end"],
        ["logits"],
        DYNAMIC,
    )

    logging.info("Quantizing %s to int8", homo_path)
    quantize_dynamic(homo_path, quantized_path, weight_type=QuantType.QInt8)
    os.replace(quantized_path, homo_path)

    embedding = torch.zeros(1, 16)

    export_onnx(
        accentor.accentor.model.stress_clf,
        (embedding,),
        os.path.join(SILERO_MODEL_DIR, "stress_clf.onnx"),
        ["emb"],
        ["logits"],
        CLF_DYNAMIC,
    )
    export_onnx(
        accentor.accentor.model.yo_clf,
        (embedding,),
        os.path.join(SILERO_MODEL_DIR, "yo_clf.onnx"),
        ["emb"],
        ["logits"],
        CLF_DYNAMIC,
    )


def export_data(accentor):
    weight = accentor.accentor.model.embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(SILERO_MODEL_DIR, "embedding.npy"), weight.astype(np.float32))

    with open(os.path.join(SILERO_MODEL_DIR, "ngram_dict.json"), "w") as f:
        json.dump(dict(accentor.accentor.model.embedding.ngram_dict), f)

    with open(os.path.join(SILERO_MODEL_DIR, "exceptions.json"), "w") as f:
        json.dump(
            {word: list(value) for word, value in accentor.accentor.exceptions.items()},
            f,
        )

    homodict = accentor.homosolver.homodict
    with open(os.path.join(SILERO_MODEL_DIR, "homodict.json"), "w") as f:
        json.dump(homodict, f)

    phrases = {
        word: pattern.pattern
        for word, pattern in accentor.homosolver.compiled_phrases.items()
    }
    with open(os.path.join(SILERO_MODEL_DIR, "phrases.json"), "w") as f:
        json.dump(phrases, f)

    tokenizer = accentor.homosolver.tokenizer
    vocab = OrderedDict(sorted(tokenizer.vocab.items(), key=lambda item: item[1]))
    with open(
        os.path.join(SILERO_MODEL_DIR, "bert_vocab.txt"), "w", encoding="utf-8"
    ) as f:
        for token in vocab:
            f.write(token + "\n")

    config = {
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
        "mask_token": tokenizer.mask_token,
        "never_split": sorted(tokenizer.never_split),
    }
    with open(os.path.join(SILERO_MODEL_DIR, "bert_config.json"), "w") as f:
        json.dump(config, f)


def main():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(SILERO_MODEL_DIR, exist_ok=True)

    accentor = load_accentor("ru")

    export_models(accentor)
    export_data(accentor)

    logging.info("silero-stress exported to %s", SILERO_MODEL_DIR)


if __name__ == "__main__":
    main()

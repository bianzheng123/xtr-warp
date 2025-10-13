import torch

from transformers import AutoTokenizer, AutoModel, logging
from huggingface_hub import hf_hub_download

def build_xtr_model():
    # NOTE Warning about unitialized decoder weights is to be expected.
    #      We only make use of the encoder anyways.
    logging.set_verbosity_error()
    model = AutoModel.from_pretrained("google/xtr-base-en", use_safetensors=True)

    tokenizer = AutoTokenizer.from_pretrained("google/xtr-base-en")

    # Source: https://huggingface.co/google/xtr-base-en/
    to_dense_path = hf_hub_download(repo_id="google/xtr-base-en", filename="2_Dense/pytorch_model.bin")

    logging.set_verbosity_warning()

build_xtr_model()

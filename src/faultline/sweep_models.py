"""Pretrained sweep inputs and offline factories. Checkpoints stay outside the repository."""

import hashlib
import json
import urllib.request
from pathlib import Path

import torch

VISION_MODELS = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnext50_32x4d",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_v2_s",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "vit_b_16",
    "swin_t",
    "convnext_tiny",
)
TEXT_MODELS = {
    "bert": "google-bert/bert-base-uncased",
    "gpt2": "openai-community/gpt2",
    "distilbert": "distilbert/distilbert-base-uncased",
    "smollm2": "HuggingFaceTB/SmolLM2-135M",
}
ORIGINAL_MODELS = (*VISION_MODELS, *TEXT_MODELS)
HELD_OUT_VISION = (
    "regnet_y_400mf",
    "densenet121",
    "shufflenet_v2_x1_0",
    "squeezenet1_1",
    "mnasnet1_0",
)
HELD_OUT_TEXT = {
    "t5_small_encoder": "google-t5/t5-small",
    "roberta_base": "FacebookAI/roberta-base",
    "opt_125m": "facebook/opt-125m",
    "pythia_160m": "EleutherAI/pythia-160m",
}
TEXT_REVISIONS = {
    # Existing SFconvertbot conversion, https://huggingface.co/facebook/opt-125m/discussions/51
    "opt_125m": "1f9886ce095904096e22b0f4d9e7ba932fa7df2a",
}
HELD_OUT_MODELS = (*HELD_OUT_VISION, *HELD_OUT_TEXT)
VISION_MODELS = (*VISION_MODELS, *HELD_OUT_VISION)
TEXT_MODELS = {**TEXT_MODELS, **HELD_OUT_TEXT}
MODELS = (*ORIGINAL_MODELS, *HELD_OUT_MODELS)
IMAGE_URLS = (
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg",
)
BOOK_URL = "https://www.gutenberg.org/ebooks/11.txt.utf-8"


class TextEncoder(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, **inputs):
        return self.inner(**inputs).last_hidden_state


def factory():
    """Only local assets are read during worker replay and standalone execution."""
    root = Path(__file__).resolve().parent
    spec = json.loads((root / "model.json").read_text(encoding="utf-8"))
    torch.manual_seed(0)
    if spec["provider"] == "torchvision":
        from torchvision.models import get_model

        model = get_model(spec["architecture"], weights=None)
    else:
        from transformers import AutoConfig, AutoModel

        configuration = dict(spec["config"])
        kind = configuration.pop("model_type")
        config = AutoConfig.for_model(kind, **configuration)
        config.use_cache = False
        if spec.get("loader") == "T5EncoderModel":
            from transformers import T5EncoderModel

            config._attn_implementation = spec["attention"]
            inner = T5EncoderModel(config)
        else:
            inner = AutoModel.from_config(
                config,
                attn_implementation=spec.get("attention", "sdpa"),
                **spec.get("model_options", {}),
            )
        model = TextEncoder(inner)
    model.load_state_dict(torch.load(root / "weights.pt", map_location="cpu", weights_only=True))
    return model.eval(), torch.load(root / "inputs.pt", map_location="cpu", weights_only=True)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path):
    path = Path(path)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(16 * 1024**2 + 1)
    if len(data) > 16 * 1024**2:
        raise ValueError("Sweep sample exceeds the 16 MiB input download limit")
    path.write_bytes(data)
    return {"url": url, "sha256": digest(path), "bytes": len(data)}


def real_inputs(directory, input_batches=1):
    if type(input_batches) is not int or input_batches < 1:
        raise ValueError("input_batches must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    sources = []
    for index, url in enumerate(IMAGE_URLS):
        sources.append(download(url, directory / f"image-{index}.jpg"))
    sources.append(download(BOOK_URL, directory / "book.txt"))
    book = (directory / "book.txt").read_text(encoding="utf-8-sig")
    start = book.index("Alice was beginning to get very tired")
    # Fixed real prose, not generated tokens. Both text samples remain in the input manifest.
    paragraphs = [
        part.strip().replace("\n", " ") for part in book[start:].split("\n\n") if part.strip()
    ]
    texts = paragraphs[:2]
    manifest = {
        "schema_version": 1,
        "sources": sources,
        "texts": texts,
        "scope": "Two real photographs and two public-domain prose paragraphs; not a representative accuracy dataset.",
    }
    if input_batches > 1:
        # Nonoverlapping source regions, not stochastic augmentations of the same pixels.
        # These are disjoint crops of two photographs, not independent photographs.
        from PIL import Image

        words = " ".join(paragraphs[2:]).split()
        if 400 * input_batches > len(words):
            raise ValueError("Insufficient disjoint source text for requested input batches")
        manifest["batches"] = []
        for batch in range(input_batches):
            images = []
            for index in range(2):
                with Image.open(directory / f"image-{index}.jpg") as image:
                    left = image.width * batch // input_batches
                    right = image.width * (batch + 1) // input_batches
                    if right <= left:
                        raise ValueError("Insufficient disjoint source pixels")
                    name = f"batch-{batch:04d}-image-{index}.png"
                    image.crop((left, 0, right, image.height)).save(directory / name)
                    images.append(
                        {
                            "file": name,
                            "sha256": digest(directory / name),
                            "source_image": index,
                            "box": [left, 0, right, image.height],
                        }
                    )
            start = batch * 400
            manifest["batches"].append(
                {
                    "batch": batch,
                    "images": images,
                    "word_range": [start, start + 400],
                    "texts": [
                        " ".join(words[start : start + 200]),
                        " ".join(words[start + 200 : start + 400]),
                    ],
                }
            )
        manifest["scope"] = (
            "Disjoint deterministic image strips from two photos and disjoint 200-word passages from one book; source dependence remains. Original full images/paragraphs are retained separately, not counted as replicates."
        )
    (directory / "inputs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_model(name, directory, inputs_directory):
    """Resolve pretrained weights once, then serialize an offline workload factory."""
    import importlib.metadata
    import shutil

    directory, inputs_directory = Path(directory), Path(inputs_directory)
    directory.mkdir(parents=True, exist_ok=False)
    input_meta = json.loads((inputs_directory / "inputs.json").read_text(encoding="utf-8"))
    if name in VISION_MODELS:
        from PIL import Image
        from torchvision.models import get_model, get_model_weights

        weights = get_model_weights(name).DEFAULT
        model = get_model(name, weights=weights).eval()
        transform = weights.transforms()
        inputs = (
            torch.stack(
                [
                    transform(Image.open(inputs_directory / f"image-{i}.jpg").convert("RGB"))
                    for i in range(2)
                ]
            ),
        )
        if "batches" in input_meta:
            inputs = [
                (
                    torch.stack(
                        [
                            transform(Image.open(inputs_directory / image["file"]).convert("RGB"))
                            for image in batch["images"]
                        ]
                    ),
                )
                for batch in input_meta["batches"]
            ]
        spec = {
            "provider": "torchvision",
            "architecture": name,
            "weights": str(weights),
            "weights_url": weights.url,
            "transform": repr(transform),
            "provider_version": importlib.metadata.version("torchvision"),
        }
    elif name in TEXT_MODELS:
        from huggingface_hub import model_info
        from transformers import AutoModel, AutoTokenizer

        repository = TEXT_MODELS[name]
        info = model_info(repository, revision=TEXT_REVISIONS.get(name))
        revision = info.sha
        files = {entry.rfilename for entry in info.siblings}
        if not files & {"model.safetensors", "model.safetensors.index.json"}:
            raise ValueError(
                "Pinned checkpoint has no safetensors weights; automatic conversion is disabled"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            repository, revision=revision, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = dict(
            tokenizer(
                input_meta["texts"],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
        )
        if "batches" in input_meta:
            inputs = [
                dict(
                    tokenizer(
                        batch["texts"],
                        padding="max_length",
                        truncation=True,
                        max_length=64,
                        return_tensors="pt",
                    )
                )
                for batch in input_meta["batches"]
            ]
        loader = AutoModel
        if name == "t5_small_encoder":
            from transformers import T5EncoderModel

            loader = T5EncoderModel
        elif name == "opt_125m":
            from transformers import AutoModelForCausalLM

            # The converted checkpoint stores the tied embedding as lm_head.weight.
            # Resolve the declared tie before extracting the pretrained base model.
            loader = AutoModelForCausalLM
        attention = "sdpa"
        model_options = {"add_pooling_layer": False} if name == "roberta_base" else {}
        inner, loading = loader.from_pretrained(
            repository,
            revision=revision,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation=attention,
            output_loading_info=True,
            **model_options,
        )
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Pretrained checkpoint did not fully initialize the model: {loading}")
        if name == "opt_125m":
            inner = inner.base_model
        inner.config.use_cache = False
        model = TextEncoder(inner).eval()
        spec = {
            "provider": "huggingface",
            "architecture": name,
            "repository": repository,
            "revision": revision,
            "revision_source": "pinned_existing_conversion"
            if name in TEXT_REVISIONS
            else "resolved_main",
            "config": inner.config.to_dict(),
            "sequence_length": 64,
            "output": "last_hidden_state",
            "attention": attention,
            "loader": "T5EncoderModel" if name == "t5_small_encoder" else "AutoModel",
            "checkpoint_loader": "AutoModelForCausalLM"
            if name == "opt_125m"
            else "T5EncoderModel"
            if name == "t5_small_encoder"
            else "AutoModel",
            "model_options": model_options,
            "use_cache": False,
            "provider_version": importlib.metadata.version("transformers"),
            "unused_checkpoint_keys": sorted(loading.get("unexpected_keys", [])),
        }
    else:
        raise ValueError(f"Unknown pretrained model: {name}")
    spec.update(
        pretrained=True,
        storage_dtype="float32",
        batch_size=2,
        input_sources=input_meta["sources"],
        catalog_set="held-out"
        if name in HELD_OUT_MODELS
        else "original"
        if name in ORIGINAL_MODELS
        else "custom",
        parameter_count=sum(p.numel() for p in model.parameters()),
    )
    torch.save(model.state_dict(), directory / "weights.pt")
    torch.save(inputs, directory / "inputs.pt")
    spec.update(
        weights_sha256=digest(directory / "weights.pt"),
        inputs_sha256=digest(directory / "inputs.pt"),
    )
    (directory / "model.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    shutil.copyfile(__file__, directory / "factory.py")
    return spec

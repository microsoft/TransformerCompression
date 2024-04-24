# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.phi.modeling_phi import PhiConfig

from .adapters.sliced_llama import SlicedLlamaConfig, SlicedLlamaForCausalLM
from .adapters.sliced_phi import SlicedPhi2Config, SlicedPhiForCausalLM
from .layernorm_fusion import fuse_modules, replace_layers
from .model_adapter import ModelAdapter, SlicingConfig
from .rotate import slice_rotated_model
from .slicing_scheduler import ConstSlicingScheduler, SlicingScheduler


def do_not_initialize(func):
    """
    A decorator that prevents initialization of torch.nn modules.
    """

    def skip(*args, **kwargs) -> None:
        pass

    def wrapper(*args, **kwargs):
        kaiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kaiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn

        return result

    return wrapper


@do_not_initialize
def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    uninitialized: bool = False,
    dtype: torch.dtype = torch.float16,
    token: str | bool | None = None,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load the model and the tokenizer from the given path.
    Set uninitialized to True when loading a pre-rotated and sliced model; in this case no weights are loaded
    in this method.
    The corresponding model adapter class must be imported before calling this method.
    Scenarios:
    - Rotate & slice HF model: model_name = name, model_path = empty, uninitialized = False
        -> Obtain the model config and weights from HF through path = name.
        -> Ignore model_path if provided.
    - Slice pre-rotated HF model: model_name = name, model_path = empty or local path, uninitialized = True
        -> Obtain the model config from HF via path = name and create uninitialized model.
        -> If the model_path is provided, confirm this use case by checking that config.json does not exist.
        -> There are no other uses of model_path in this case.
    - Rotate & slice local model: model_name = name, model_path = local path, uninitialized = False
        -> Obtain the model config through path, and the pretrained weights from the local path.
        -> Use the model name only to determine the correct model adapter to use.
    - Slice pre-rotated local model: model_name = name, model_path = local path, uninitialized = True
        -> Obtain the model config from the local path and create an uninitialized model.
        -> Use the model name only to determine the correct model adapter to use.
        -> Confirm this case by checking that config.json exists.
    """
    model_type = "uninitialized" if uninitialized else "pretrained"
    local_model = model_path is not None

    if local_model and uninitialized:
        local_model = (pathlib.Path(model_path) / "config.json").exists()

    # for HF models the path to use is the model name
    if not local_model:
        model_path = model_name

    logging.info(
        f"Loading %s config %s from %s",
        model_name,
        "and model weights" if not uninitialized else "",
        model_path if local_model else 'Hugging Face',
    )

    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=model_path,
        model_type=model_type,
        dtype=dtype,
        local_files_only=local_model,
        token=token,
    )

    model = model_adapter.model
    model.seqlen = model.config.max_position_embeddings
    model.eval()  # This switches off dropout.
    model_adapter.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token, local_files_only=local_model)

    model_adapter.post_init(tokenizer)
    logging.info("Loading model done")

    return model_adapter, tokenizer


def load_sliced_model(
    model_name: str,
    sliced_model_path: str,
    *,
    token: str | None = None,
    lora_config: LoraConfig | None = None,
    sparsity: float | None = None,
    round_interval: int | None = 1,
) -> tuple[ModelAdapter | torch.nn.Module, PreTrainedTokenizerBase]:
    """
    Load the sliced model and the tokenizer from the given path. If lora_config is supplied as an arg then this
    function will return a PEFT model (post-slicing finetuned model).
    The corresponding model adapter class must be imported before calling this method.
    """

    model_adapter, tokenizer = get_model_and_tokenizer(
        model_name,
        model_path=sliced_model_path,
        uninitialized=True,
        token=token,
    )

    # handle loading sliced HF compatible models
    if model_name.startswith("microsoft") or model_name.startswith("llama"):
        new_embedding_dimension = int((1 - sparsity) * model_adapter.hidden_size)
        new_embedding_dimension -= new_embedding_dimension % round_interval

        scheduler = ConstSlicingScheduler(new_embedding_dimension)
        
        layers = model_adapter.get_layers()
        scheduler.setup(
            hidden_size=model_adapter.hidden_size,
            layers_num=len(layers),
            parallel_blocks=True,
        )

        sliced_model = SlicedPhiForCausalLM.from_pretrained(
            sliced_model_path,
            scheduler=scheduler,
            config_path=sliced_model_path,
            sparsity=sparsity,
            new_hidden_size=new_embedding_dimension,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token, local_files_only=True)
        return sliced_model, tokenizer

    my_model_suffix = pathlib.Path(model_name).name
    my_sliced_model_name = f"{my_model_suffix}_{sparsity}.pt"
    my_sliced_model_config = f"{my_model_suffix}_{sparsity}.json"

    replace_layers(model_adapter)
    fuse_modules(model_adapter)

    hidden_size = model_adapter.hidden_size
    for layer_adapter in model_adapter.get_layers():
        if not model_adapter.parallel_blocks:
            layer_adapter.layer.mlp_shortcut_Q = torch.nn.Parameter(
                torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
            )
        layer_adapter.layer.attn_shortcut_Q = torch.nn.Parameter(
            torch.zeros(hidden_size, hidden_size).to(dtype=torch.float16)
        )

    config_path = pathlib.Path(sliced_model_path) / my_sliced_model_config

    if config_path.exists():
        model_adapter.slicing_conf = SlicingConfig.from_json_string(config_path.read_text())

    if model_adapter.slicing_conf is None:
        # assume the model was sliced with the const sparsity specified in the arguments to this method
        new_embedding_dimension = int((1 - sparsity) * hidden_size)
        new_embedding_dimension -= new_embedding_dimension % round_interval
        config = SlicingConfig()
        config.const_dimension = new_embedding_dimension
        model_adapter.slicing_conf = config

    slice_rotated_model(model_adapter)

    if lora_config:
        model_adapter.model = get_peft_model(model_adapter.model, lora_config)

    logging.info(f"Loading sliced model weights from {sliced_model_path}")
    model_adapter.model.load_state_dict(
        torch.load(str(pathlib.Path(sliced_model_path) / my_sliced_model_name), map_location="cpu")
    )
    model_adapter.model.eval()

    return model_adapter, tokenizer


def save_sliced_model(
    model_name: str,
    dtype: torch.dtype,
    model: torch.nn.Module,
    scheduler: SlicingScheduler,
    save_sliced_model_dir: str | pathlib.Path,
    sparsity: float,
    new_hidden_size: int,
    slicing_conf: SlicingConfig,
):
    if model_name == "microsoft/phi-2":
        config = PhiConfig.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

        config.save_pretrained("phi_config")
        config_to_save = SlicedPhi2Config.from_pretrained(
            config_path="phi_config", sparsity=sparsity, new_hidden_size=new_hidden_size
        )

        sliced_model = SlicedPhiForCausalLM(config_to_save, scheduler).to(dtype)
        sliced_model.load_state_dict(model.state_dict(), strict=True, assign=True)
        sliced_model.save_pretrained(save_sliced_model_dir)
        
        sliced_model = SlicedPhiForCausalLM.from_pretrained(
            save_sliced_model_dir,
            scheduler=scheduler,
            config_path=save_sliced_model_dir,
            sparsity=sparsity,
            new_hidden_size=new_hidden_size,
        )

    elif "meta-llama" in model_name:
        config = LlamaConfig.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        config.save_pretrained("llama_config")
        config_to_save = SlicedLlamaConfig.from_pretrained(
            config_path="llama_config",
            sparsity=sparsity,
            new_hidden_size=new_hidden_size,
        )

        sliced_model = SlicedLlamaForCausalLM(config_to_save, scheduler).to(dtype)
        sliced_model.load_state_dict(model.state_dict(), strict=True, assign=True)
        sliced_model.save_pretrained(save_sliced_model_dir)
    else:
        # Save the sliced model for other models types
        sliced_model_name = save_sliced_model_dir / f'{pathlib.Path(model_name).name}_{sparsity}.pt'
        torch.save(model.state_dict(), sliced_model_name)
        config_path = sliced_model_name.with_suffix('.json')
        config_path.write_text(slicing_conf.to_json_string())

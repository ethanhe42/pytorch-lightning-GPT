import functools
import warnings
from typing import Any, Optional, Tuple
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch
import torch.optim
from lightning import LightningModule
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning_utilities.core.overrides import is_overridden


import nanogpt.model

MINGPT_PRESETS = {
    # names follow the huggingface naming conventions
    # GPT-1
    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
    # GPT-2 configs
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
    "gpt2-xxl": dict(n_layer=96, n_head=25, n_embd=1600),  # 2951M params
    "gpt2-xxxl": dict(n_layer=100, n_head=30, n_embd=1920),  # 4426M params
    "gpt2-4xl": dict(n_layer=190, n_head=30, n_embd=1920),  # 8409M params
    # Gophers
    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
    # (there are a number more...)
    # I made these tiny models up
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
}


class NanoGPT(LightningModule):
    nanogpt: nanogpt.model.GPT

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        model_type: Optional[str] = "gpt2",
        n_layer: Optional[int] = None,
        n_head: Optional[int] = None,
        n_embd: Optional[int] = None,
        dropout: float = 0.1,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.build_nanogpt_configs()
        if not is_overridden("configure_sharded_model", self, LightningModule):
            self.nanogpt = nanogpt.model.GPT(self.nanogpt_config)

    def build_nanogpt_configs(self) -> None:
        params = [
            self.hparams.n_layer,
            self.hparams.n_head,
            self.hparams.n_embd,
        ]

        params_given = all([el is not None for el in params])
        some_params_given = any([el is not None for el in params])

        if some_params_given and not params_given:
            raise ValueError(
                "Please provide all values for n_layer, n_head, and n_embd, or just model_type."
                f"Got n_layer={self.hparams.n_layer}, n_head={self.hparams.n_head}, "
                f"and n_embd={self.hparams.n_embd}."
            )

        if not params_given:
            # We take ownership of presets over minGPT here
            preset = MINGPT_PRESETS[self.hparams.model_type]
            self.hparams.update(preset)
            self.hparams.model_type = None

        self.nanogpt_config = nanogpt.model.GPTConfig()
        self.merge_with_hparams(self.nanogpt_config)

        C = CfgNode()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        self.nanogpt_trainer_config = C
        
        self.merge_with_hparams(self.nanogpt_trainer_config)

    def merge_with_hparams(self, config) -> None:
        for k, v in self.hparams.items():
            if hasattr(config, k):
                setattr(config, k, v)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.nanogpt(idx, targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.nanogpt.configure_optimizers(
            weight_decay=self.nanogpt_trainer_config.weight_decay,
            learning_rate=self.nanogpt_trainer_config.learning_rate,
            betas=self.nanogpt_trainer_config.betas,
            device_type=self.hparams.device_type,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("train_loss", loss)
        return loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None
    ) -> torch.Tensor:
        return self.nanogpt.generate(idx, max_new_tokens, temperature, top_k)

class DeepSpeedNanoGPT(NanoGPT):
    # TODO: activation checkpointing (requires overriding forward)
    def __init__(self, fused_adam: bool = True, offload: bool = False, **kwargs: Any):
        if fused_adam and offload:
            raise RuntimeError(
                "Cannot use FusedAdam and CPUAdam at the same time! "
                "Please set either `fused_adam` or `offload` to False."
            )

        kwargs["device_type"] = "cuda" if fused_adam or kwargs.pop("device_type", "cpu") == "cuda" else "cpu"

        super().__init__(**kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = super().configure_optimizers()

        return _get_deepspeed_optimizer(
            optimizer,
            fused_adam=self.hparams.device_type == "cuda",
            cpu_offload=self.hparams.offload,
            learning_rate=self.hparams.learning_rate,
            betas=self.hparams.betas,
        )

    def configure_sharded_model(self) -> None:
        self.nanogpt = nanogpt.model.GPT(self.nanogpt_config)


class FSDPNanoGPT(NanoGPT):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        _register_gpt_strategy()

    def configure_optimizers(self) -> torch.optim.AdamW:
        assert isinstance(self.trainer.model, torch.nn.Module)
        return _get_fsdp_optimizers(
            self.trainer.model,
            weight_decay=self.nanogpt_trainer_config.weight_decay,
            learning_rate=self.nanogpt_trainer_config.learning_rate,
            betas=self.nanogpt_trainer_config.betas,
        )


def _register_gpt_strategy() -> None:
    from lightning.pytorch.strategies import StrategyRegistry
    from lightning.pytorch.strategies.fsdp import FSDPStrategy
    from torch.distributed.fsdp import BackwardPrefetch
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if "fsdp-gpt" in StrategyRegistry.available_strategies():
        return

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={nanogpt.model.Block, mingpt.model.Block}
    )

    StrategyRegistry.register(
        name="fsdp-gpt",
        strategy=FSDPStrategy,
        description="FSDP strategy with memory optimizations enabled for GPT large scale pretraining.",
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=[nanogpt.model.Block, mingpt.model.Block],
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )


def _get_deepspeed_optimizer(
    optimizer: torch.optim.Optimizer,
    cpu_offload: bool,
    fused_adam: bool,
    learning_rate: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    optim_groups = optimizer.param_groups

    # import locally because of https://github.com/Lightning-AI/lightning/pull/15610
    if cpu_offload and _DEEPSPEED_AVAILABLE:
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        return DeepSpeedCPUAdam(optim_groups, lr=learning_rate, betas=betas)

    elif fused_adam and _DEEPSPEED_AVAILABLE:
        from deepspeed.ops.adam import FusedAdam

        return FusedAdam(optim_groups, lr=learning_rate, betas=betas)

    elif fused_adam or cpu_offload:
        warnings.warn(
            "Deepspeed is not available, so cannot enable fused adam or cpu offloaded adam. Please install deepspeed!"
        )

    return optimizer


def _get_fsdp_optimizers(
    model: torch.nn.Module, learning_rate: float, weight_decay: float, betas: Tuple[float, float]
) -> torch.optim.AdamW:
    # fsdp only supports a single parameter group and requires the parameters from the already wrapped model
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)


class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)

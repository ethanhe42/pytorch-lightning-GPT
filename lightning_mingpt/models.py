import math
import functools

import lightning as L

import torch.nn as nn
from torch.optim import AdamW

import mingpt.model
import mingpt.trainer

from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning.pytorch.utilities.model_helpers import is_overridden


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


class GPT(L.LightningModule):
    mingpt: nn.Module

    def __init__(
        self,
        vocab_size,
        block_size,
        model_type="gpt2",
        n_layer=None,
        n_head=None,
        n_embd=None,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        weight_decay=0.1,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.build_mingpt_configs()
        if not is_overridden("configure_sharded_model", self, L.LightningModule):
            self.mingpt = mingpt.model.GPT(self.mingpt_config)

    def build_mingpt_configs(self):
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

        self.mingpt_config = mingpt.model.GPT.get_default_config()
        self.merge_with_hparams(self.mingpt_config)

        self.mingpt_trainer_config = mingpt.trainer.Trainer.get_default_config()
        self.merge_with_hparams(self.mingpt_trainer_config)

    def merge_with_hparams(self, config):
        keys = set(config.to_dict().keys())
        hparams = {k: v for k, v in self.hparams.items() if k in keys}
        config.merge_from_dict(hparams)

    def forward(self, idx, targets=None):
        return self.mingpt(idx, targets)

    def configure_optimizers(self):
        return self.mingpt.configure_optimizers(self.mingpt_trainer_config)

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("train_loss", loss)
        return loss

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        return self.mingpt.generate(idx, max_new_tokens, temperature, do_sample, top_k)


class DeepSpeedGPT(GPT):
    # TODO: activation checkpointing (requires overriding forward)
    def __init__(self, fused_adam: bool = True, offload: bool = False, **kwargs):
        if fused_adam and offload:
            raise RuntimeError('Cannot use FusedAdam on cpu! Please choose one of them.')

        super().__init__(**kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        optim_groups = optimizer.param_groups

        # import locally because of https://github.com/Lightning-AI/lightning/pull/15610
        if self.hparams.offload and _DEEPSPEED_AVAILABLE:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            return DeepSpeedCPUAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

        elif self.hparams.fused_adam and _DEEPSPEED_AVAILABLE:
            from deepspeed.ops.adam import FusedAdam
            return FusedAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

        elif self.hparams.fused_adam or self.hparams.offload:
            warnings.warn('Deepspeed is not available, so cannot enable fused adam or cpu offloaded adam. Please install deepspeed!')

        return optimizer

    def configure_sharded_model(self) -> None:
        self.mingpt = mingpt.model.GPT(self.mingpt_config)


class FSDPGPT(GPT):
    def __init__(self, offload=False, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._register_gpt_strategy()


    def configure_optimizers(self):
        optimizer = self.mingpt.configure_optimizers(self.mingpt_trainer_config, model=self.trainer.model, multiple_optim_groups=False)
        optim_groups = optimizer.param_groups

        return AdamW(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

    @staticmethod
    def _register_gpt_strategy():
        from lightning.pytorch.strategies import StrategyRegistry
        from lightning.pytorch.strategies.fully_sharded_native import DDPFullyShardedNativeStrategy
        from torch.distributed.fsdp import BackwardPrefetch
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={mingpt.model.Block})
        StrategyRegistry.register(
            name="fsdp-gpt",
            strategy=DDPFullyShardedNativeStrategy,
            description="FSDP strategy with memory optimizations enabled for GPT large scale pretraining.",
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=[mingpt.model.Block],
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        )

                


import math

import lightning as L

import torch.nn as nn

import mingpt.model
import mingpt.trainer

from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning.pytorch.utilities.model_helpers import is_overridden

if _DEEPSPEED_AVAILABLE:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

_XFORMERS_AVAILABLE = False
try:
    import xformers.factory as xformers

    _XFORMERS_AVAILABLE = True
except ImportError:
    pass


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
        if not is_overridden('configure_sharded_model', self, L.LightningModule):
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

    def configure_optimizers(self):
        return self.mingpt.configure_optimizers(self.mingpt_trainer_config)

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        _, loss = self.mingpt(idx, targets)
        self.log("train_loss", loss)
        return loss

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        return self.mingpt.generate(idx, max_new_tokens, temperature, do_sample, top_k)


class DeepSpeedGPT(GPT):
    # TODO: activation checkpointing (requires overriding forward)
    def __init__(self, offload=False, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        optim_groups = optimizer.param_groups

        if self.hparams.offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

        return FusedAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

    def configure_sharded_model(self) -> None:
        self.mingpt = mingpt.model.GPT(self.mingpt_config)


class _XFormersMinGPT(mingpt.model.GPT):
    def __init__(
        self,
        mingpt_config,
        attention="scaled_dot_product",
        feedforward="mlp",
        mlp_pdrop=0.1,
        hidden_layer_multiplier=4,
    ):
        # We skip super().__init__() to avoid allocating weights for minGPT
        # and instead let xformers do it
        nn.Module.__init__(self)

        ffwd = dict(mlp="MLP", fusedmlp="FusedMLP")[feedforward]

        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": mingpt_config.n_layer,
                "dim_model": mingpt_config.n_embd,
                "residual_norm_style": "post",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": mingpt_config.block_size,
                    "vocab_size": mingpt_config.vocab_size,
                },
                "multi_head_config": {
                    "num_heads": mingpt_config.n_head,
                    "residual_dropout": mingpt_config.resid_pdrop,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": attention,
                        "dropout": mingpt_config.attn_pdrop,
                        "causal": True,
                        "seq_len": mingpt_config.block_size,
                        "num_rules": mingpt_config.n_head,
                    },
                },
                "feedforward_config": {
                    "name": ffwd,
                    "dropout": mlp_pdrop,
                    "activation": "gelu",
                    "hidden_layer_multiplier": hidden_layer_multiplier,
                },
            }
        ]

        config = xformers.xFormerConfig(xformer_config)
        config.weight_init = "small"

        transformer = xformers.xFormer.from_config(config)

        ln_f = nn.LayerNorm(mingpt_config.n_embd)
        lm_head = nn.Linear(mingpt_config.n_embd, mingpt_config.vocab_size, bias=False)

        self.transformer = transformer
        self.lm_head = lm_head

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * mingpt_config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")


class XFormersGPT(GPT):
    def __init__(
        self, attention="scaled_dot_product", mlp_pdrop=0.1, hidden_layer_multiplier=4, feedforward="mlp", **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.build_mingpt_configs()

        self.mingpt = _XFormersMinGPT(self.mingpt_config)

import math

import lightning as L

import torch.nn as nn

import mingpt.model
import mingpt.trainer

from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE

if _DEEPSPEED_AVAILABLE:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

_XFORMERS_AVAILABLE = True
try:
    import xformers

    _XFORMERS_AVAILABLE = True
except ImportError:
    pass


class GPT(L.LightningModule):
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
        self.mingpt = mingpt.model.GPT(self.config)

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

        if params_given:
            self.hparams.model_type = None

        self.config = mingpt.model.GPT.get_default_config()
        self.merge_with_hparams(self.config)

        self.optimizer_config = mingpt.trainer.Trainer.get_default_config()
        self.merge_with_hparams(self.optimizer_config)

    def merge_with_hparams(self, config):
        keys = set(config.to_dict().keys())
        hparams = {k: v for k, v in self.hparams.items() if k in keys}
        config.merge_from_dict(hparams)

    def configure_optimizers(self):
        return self.mingpt.configure_optimizers(self.optimizer_config)

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


class _XFormersMinGPT(mingpt.model.GPT):
    def __init__(
        self,
        config,
        attention="scaled_dot_product",
        mlp_pdrop=0.1,
        hidden_layer_multiplier=4,
    ):
        # A list of the encoder or decoder blocks which constitute the Transformer.
        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": config.n_layer,
                "dim_model": config.n_embd,
                "residual_norm_style": "post",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": config.block_size,
                    "vocab_size": config.vocab_size,
                },
                "multi_head_config": {
                    "num_heads": config.n_head,
                    "residual_dropout": config.resid_pdrop,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": attention,
                        "dropout": config.attn_pdrop,
                        "causal": True,
                        "seq_len": config.block_size,
                        "num_rules": config.n_head,
                    },
                },
                "feedforward_config": {
                    "name": "FusedMLP",  # Use MLP if Triton is not available
                    "dropout": mlp_pdrop,
                    "activation": "gelu",
                    "hidden_layer_multiplier": hidden_layer_multiplier,
                },
            }
        ]

        config = xformer.xFormerConfig(xformer_config)
        config.weight_init = "small"

        transformer = xformer.xFormer.from_config(config)

        ln_f = nn.LayerNorm(config.n_embd)
        lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer = transformer
        self.lm_head = lm_head

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")


class XFormersGPT(GPT):
    def __init__(self, attention="scaled_dot_product", mlp_pdrop=0.1, hidden_layer_multiplier=4, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.build_mingpt_configs()

        self.mingpt = _XFormersMinGPT(self.config)

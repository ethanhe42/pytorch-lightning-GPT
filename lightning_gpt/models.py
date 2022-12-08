import lightning as L

import mingpt.model
import mingpt.trainer

from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.strategies.deepspeed import _DEEPSPEED_AVAILABLE

if _DEEPSPEED_AVAILABLE:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


class GPT(L.LightningModule):
    def __init__(self,
        vocab_size,
        block_size,
        model_type = "gpt2",
        n_layer = None,
        n_head = None,
        n_embd = None,
        embd_pdrop = 0.1,
        resid_pdrop = 0.1,
        attn_pdrop = 0.1,
        weight_decay = 0.1,
        learning_rate = 3e-4,
        betas = (0.9, 0.95)
    ):
        super().__init__()
        self.save_hyperparameters()

        params_given = all([n_layer is not None, n_head is not None, n_embd is not None])
        some_params_given = any([n_layer is not None, n_head is not None, n_embd is not None])

        if some_params_given and not params_given:
            raise ValueError(
                "Please provide all values for n_layer, n_head, and n_embd, or just model_type."
                f"Got n_layer={n_layer}, n_head={n_head}, and n_embd={n_embd}."
            )

        if params_given:
            self.hparams.model_type = None

        self.config = mingpt.model.GPT.get_default_config()
        self.merge_with_hparams(self.config)

        self.optimizer_config = mingpt.trainer.Trainer.get_default_config()
        self.merge_with_hparams(self.optimizer_config)

        self.mingpt = mingpt.model.GPT(self.config)

    def merge_with_hparams(self, config):
        keys = set(config.to_dict().keys())
        hparams = {k: v for k, v in self.hparams.items() if k in keys}
        config.merge_from_dict(hparams)

    def configure_optimizers(self):
        return self.mingpt.configure_optimizers(self.optimizer_config)

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        _, loss = self.mingpt(idx, targets)
        self.log('train_loss', loss)
        return loss

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        return self.mingpt.generate(idx, max_new_tokens, temperature, do_sample, top_k)


class DeepSpeedGPT(GPT):
    # TODO: activation checkpointing (requires overriding forward)

    def is_strategy_deepspeed(self):
        strategy = self.trainer.strategy
        return isinstance(strategy, DeepSpeedStrategy)

    def is_strategy_deepspeed_offload(self):
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            config = strategy.config['zero_optimization']
            return config.get('offload_optimizer') or config.get('offload_param')
        return False

    def setup(self):
        self.use_deepspeed = self.is_strategy_deepspeed()
        self.use_deepspeed_offload = self.is_strategy_deepspeed_offload()

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        optim_groups = optimizer.param_groups

        if self.use_deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

        if self.use_deepspeed:
            return FusedAdam(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)

        return optimizer


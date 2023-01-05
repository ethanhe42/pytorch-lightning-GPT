import lightning as L
import torch

import mingpt
from lightning_mingpt import models


def test_mingpt_vs_lightning_mingpt():
    vocab_size = 65
    block_size = 128
    model_type = "gpt-mini"

    x = torch.randint(0, vocab_size, (1, 12))

    mingpt_config = mingpt.model.GPT.get_default_config()
    mingpt_config.vocab_size = vocab_size
    mingpt_config.block_size = block_size
    mingpt_config.model_type = model_type

    mingpt_model = mingpt.model.GPT(mingpt_config)

    lit_model = models.MinGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        model_type=model_type
    )

    for target_param, param in zip(lit_model.parameters(), mingpt_model.parameters()):
        target_param.data.copy_(param.data)

    mingpt_model.eval()
    lit_model.eval()

    mingpt_y, _ = mingpt_model(x)
    lit_y, _ = lit_model(x)

    torch.testing.assert_close(mingpt_y, lit_y)


def test_nanogpt_vs_lightning_nanogpt():
    vocab_size = 65
    block_size = 128
    model_type = "gpt-mini"

    x = torch.randint(0, vocab_size, (1, 12))

    mingpt_config = nano.model.GPTConfig()
    mingpt_config.vocab_size = vocab_size
    mingpt_config.block_size = block_size
    mingpt_config.model_type = model_type

    nanogpt_model = nano.model.GPT(mingpt_config)

    lit_model = models.NanoGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        model_type=model_type
    )

    for target_param, param in zip(lit_model.parameters(), mingpt_model.parameters()):
        target_param.data.copy_(param.data)

    nanogpt_model.eval()
    lit_model.eval()

    nanogpt_y, _ = nanogpt_model(x)
    lit_y, _ = lit_model(x)

    torch.testing.assert_close(nanogpt_y, lit_y)

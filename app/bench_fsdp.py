#! pip install -U --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu117
#! pip install git+https://github.com/Lightning-AI/lightning-minGPT
#! curl https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt --create-dirs -o ${HOME}/data/input.txt -C -

import os

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning_mingpt import data, models, callbacks, bench


class FSDPGPTBench(bench.Bench):
    def __init__(self, num_runs, *args, **kwargs):
        super().__init__(num_runs, *args, **kwargs)
        self.num_workers = 4
        self.batch_size = 64
        self.max_epochs = 10
        self.precision = 16
        self.model_type = "gpt2"

    def create(self):
        torch.set_float32_matmul_precision("high")

        with open(os.path.expanduser("~/data/input.txt")) as f:
            text = f.read()

        dataset = data.CharDataset(text, block_size=128)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        model = models.FSDPGPT(
            vocab_size=dataset.vocab_size,
            block_size=dataset.block_size,
            model_type=self.model_type,
        )

        return model, dataloader

    def train(self, model, dataloader):
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=1.0,
            callbacks=callbacks.CUDAMetricsCallback(),
            accelerator="cuda",
            devices="auto",
            precision=self.precision,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            replace_sampler_ddp=False,
            strategy="fsdp_native",
        )

        trainer.fit(model, dataloader)

    def run(self):
        model, dataloader = self.create()

        self.run_benchmark(
            self.train,
            args=(model, dataloader),
            num_runs=10
        )

        model, dataloader = self.create()
        model = torch.compile(model)

        self.run_benchmark(
            self.train,
            args=(model, dataloader),
            num_runs=10
        )


app = L.LightningApp(
    L.app.components.LightningTrainerMultiNode(
        FSDPGPTBench,
        num_nodes=2,
        cloud_compute=L.CloudCompute("gpu-fast"),
    )
)

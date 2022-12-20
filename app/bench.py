#! pip install light-the-torch
#! ltt install --pytorch-channel nightly torch --upgrade 
#! pip install git+https://github.com/Lightning-AI/lightning-minGPT@bench --upgrade
#! curl https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt --create-dirs -o ${HOME}/data/input.txt -C -

import os

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning_mingpt import data, models, callbacks, bench


class GPTBench(bench.Bench):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = 4
        self.batch_size = 64
        self.max_epochs = 5
        self.precision = 16
        self.model_type = "gpt2"
        self.num_runs = 5

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

        model = models.GPT(
            vocab_size=dataset.vocab_size,
            block_size=dataset.block_size,
            model_type=self.model_type,
        )

        return model, dataloader

    def train(self, model, dataloader):
        trainer = L.Trainer(
            fast_dev_run=True,
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
        )

        trainer.fit(model, dataloader)
        final_loss = trainer.fit_loop.running_loss.last().item()
        return final_loss

    def run(self):
        model, dataloader = self.create()

        self.run_benchmark(
            name="nocompile",
            fn=self.train,
            args=(model, dataloader),
            num_runs=self.num_runs
        )

        model, dataloader = self.create()
        model = torch.compile(model)

        self.run_benchmark(
            "compile",
            self.train,
            args=(model, dataloader),
            num_runs=self.num_runs
        )


app = L.LightningApp(
    GPTBench(cloud_compute=L.CloudCompute("gpu-fast")),
)

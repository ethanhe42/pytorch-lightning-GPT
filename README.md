# lightning-GPT

TPU is supported!

lightning-GPT is a minimal wrapper around Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT) and [nanoGPT](https://github.com/karpathy/nanoGPT) in Lightning.

It is aimed at providing a minimal Lightning layer on top of minGPT and nanoGPT, while leveraging the full breadth of Lightning.

There are currently a few options:

- `NanoGPT`: the GPT model from nanoGPT vanilla (set `--implementation=nanogpt`)
- `DeepSpeedMinGPT`: the GPT model from minGPT made DeepSpeed-ready (set `--strategy=deepspeed`)
- `DeepSpeedNanoGPT`: the GPT model from nanoGPT made DeepSpeed-ready (set `--strategy=deepspeed`)
- `FSDPMinGPT`: the GPT model from minGPT made FSDP (native)-ready (set `--strategy=fsdp-gpt`)
- `FSDPNanoGPT`: the GPT model from nanoGPT made FSDP (native)-ready (set `--strategy=fsdp-gpt`)

minGPT and nanoGPT are vendored with the repo in the `mingpt` and `nanogpt` directories respectively. Find the respective LICENSE there.

Thanks to:

- @karpathy for the original minGPT and nanoGPT implementation
- @williamFalcon for the first Lightning port
- @SeanNaren for the DeepSpeed pieces

## Installation

```shell
git submodule update --init --recursive
pip install -e .
```

## NanoGPT

First install the dependencies.

```shell
pip install -r requirements.txt
pip install -r requirements/nanogpt.txt
```

then

```shell
python train.py
```

See

```shell
python train.py --help
```

for the available flags.

## DeepSpeed

Install the extra-dependencies:

```shell
pip install -r requirements/deepspeed.txt
```

and pass the `strategy` flag to the script

```shell
python train.py --implementation mingpt --strategy deepspeed
```

or

```shell
python train.py --implementation nanogpt --strategy deepspeed
```

## FSDP native

Pass the `strategy` flag to the script

```shell
python train.py --implementation mingpt --strategy fsdp_native
```

or

```shell
python train.py --implementation nanogpt --strategy fsdp_native
```

## PyTorch 2.0

To run on dynamo/inductor from the PyTorch 2.0 compiler stack, run

```shell
python train.py --compile dynamo
```

Note that you will need a recent `torch` nightly (1.14.x) for `torch.compile`
to be available.

## Credits

- https://github.com/karpathy/nanoGPT
- https://github.com/karpathy/minGPT
- https://github.com/SeanNaren/minGPT
- https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html

## License

Apache 2.0 license https://opensource.org/licenses/Apache-2.0

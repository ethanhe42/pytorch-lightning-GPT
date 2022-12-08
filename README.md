# lightning-gpt

lightning-gpt is an implementation of Andrej Karpathy's minGPT in Lightning.

It is aimed at providing a minimal Lightning layer on top of minGPT, while allowing to fully leverage Lightning functionality.

There are currently two options:

* GPT: the GPT model from minGPT vanilla
* DeepSpeedGPT: the GPT model from minGPT made DeepSpeed-ready

minGPT is vendored with the repo in the `mingpt` directory. Find the LICENSE for minGPT there.

Thanks to @karpathy for the original minGPT implementation and @SeanNaren for the DeepSpeed pieces.

To run the example, first install the dependencies

```
pip install -r requirements.txt
```

then

```
python train.py
```

See

```
python train.py --help
```

for the available flags.

For DeepSpeed, install the extra-dependencies:

```
pip install -r requirements-deepspeed.txt
```

## License

Apache 2.0 license https://opensource.org/licenses/Apache-2.0

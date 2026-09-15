## When defining the primary config.yaml file inside the configs/ directory I tried adding the loss parameter under the defaults key, but it didn't work because the loss config is present as a subset of model config

Alright so consider the baseline.yaml file in configs/model/ directory:
```bash
defaults:
  - network: elitnet
  - loss: cross_entropy
  - optimizer: adam
  - lr_scheduler: warmup

name: ELiTNet
```

Now, the folder tree for the configs/model/ directory looks something like:

model
----loss
----lr_scheduler
----optimizer
----network
baseline.yaml

So notice how baseline.yaml is sort of defining the opitons of all the sub configs in the model folder

#### Also when we run the script uv train.py, hydra first sanity checks the pipeline by trying to do a forward pass in eval mode
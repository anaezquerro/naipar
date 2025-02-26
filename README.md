# Better Benchmarking LLMs for Zero-Shot Parsing :mushroom: 

Hi :wave:! This is a Python implementation of the uninformed baselines and zero-shot parsers used in our paper :memo:[Better Benchmarking LLMs for Zero-Shot Parsing](https://dspace.ut.ee/items/a8ed5397-ee52-4f32-a86d-a130e9926cbc).

## Installation 

This code was tested in [Python 3.8.20](https://www.python.org/downloads/release/python-380/) with [PyTorch 2.3.1](https://pytorch.org/get-started/previous-versions/), [Transformers 4.45.2](https://pypi.org/project/transformers/) and NVIDIA 535-560 and CUDA 12.4-12.6. We provide in this repository the [environment.yaml](environment.yaml) file to create the Anaconda environment and the [requirements.txt](requirements.txt) to check the versions of the libraries needed to run our code.

```shell 
conda env create -f environment.yaml
```

or 

```shell 
python3.8 -m pip install -r requirements.txt
```

Some uninformed baselines require the [Linear Arrangement Library (LAL)](https://cqllab.upc.edu/lal/), so you might need to follow the installation steps from the [official repository](https://github.com/LAL-project/python-interface/). We also provided in the [docs](docs/) folder the guides to this library. For simplicity, we provide at [env.tar.gz](https://drive.google.com/file/d/1R15IaJ5NB2V82xQ6SVeplXJ7F65DcZo8/view?usp=sharing) a compressed file with the Miniconda environment, so you only need to unzip it and locate it at your [~/miniconda3/envs/](~/miniconda3/envs/) folder.

## Reproducibility


### Uninformed baselines

Uninformed baselines can be executed from the [run.py](run.py) script introducing in the first argument the name of the basleine. For example, to run the left-branching baseline use:

```shell 
python3 run.py left --path results/left/ \  
    --data treebanks/english-ewt/test.conllu --ref treebanks/english-ewt/train.conllu
```
The name fo the baseline can be modified to `right` (for the right-branching baseline), `random` to generate uniformly random trees, `random-proj` for `random` generation but forcing projectivity, `linear` (`linear-proj`) for optimal (projective) linear arrangement, and `sample` for treebank sampling.

The argument `path` must specify a folder where the final metric and predicted CoNLL file will be stored. The `data` argument is used for the path to test set that is evaluated and the `ref` indicates the path of the reference (training) set.

## Zero-shot parsers 

LLM-based zero-shot dependency parsers are also executed from the [run.py](run.py) script. The first argument must be `zero` to activate the zero-shot prediction, and the `pretrained` argument should use a HuggingFace model. Additionally, the `precision`,  `batch-size`, `load4bit` and `temperature` can be configured. For example, to run a zero-shot parser based on [Llama-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), use:

```shell

python3 run.py zero --pretrained meta-llama/Llama-3.2-3B-Instruct --path results/zero \
    --data treebanks/english-ewt/test.conllu --ref treebanks/english-ewt/train.conllu 
```

In the `path` folder, three different CoNLL files and two metrics (.pickle files) will be stored. The `gold.conllu` file, which containts the parsed gold CoNLL (where longer sentences than the maximum context of the LLM are skipped), the predicted CoNLL after the first postprocessing step and the predicted CoNLL after the second postprocessing step. Inside the `path` folder, the logs of the LLM will be also stored in the `log-output` subfolder.



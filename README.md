[![JetBrains Research](https://jb.gg/badges/research.svg)](https://research.jetbrains.org/groups/ml_methods)
[![Linux, MacOS, Docker](https://github.com/JetBrains-Research/similar-repositories/workflows/Linux,%20MacOS,%20Docker/badge.svg)](https://github.com/JetBrains-Research/similar-repositories/actions?query=workflow%3A%22Linux%2C+MacOS%2C+Docker%22)

# Similar projects search

An embedding-based approach to detect simliar repositories on GitHub.

## How it works

We took a dataset of 9 million GitHub repositories from the [paper](https://arxiv.org/pdf/1704.00135.pdf) by Markovtsev et al.
It contains all the repositories on GitHub as of the end of 2016, excluding explicit and implicit forks (see paper for the details).

Then, we computed embeddings for tokens in code with [fasttext](https://github.com/facebookresearch/fastText) and clustered them
into 256 groups of similar tokens. The clusters represent topics at token level. For each repository we computed 
its numerical representation in the space of token clusters. It is a 256-dimensional vector, where coordinate along the
dimension _K_ is the number of occurrences of tokens from cluster _K_ in the repository.

Finally, we measure similarity of two repositories as cosine distance between their vector representations. The intuition
behind this idea is that for repositories with similar topics the distribution of token-topics should also be similar.

## Getting started

To run the tool, clone this project 
```
git clone https://github.com/JetBrains-Research/similar-repositories.git
```

### Run the project from source (Linux & macOS)

#### Pip users
* Install required dependencies

```shell script
pip install cython
pip install -r requirements.txt
```

#### Conda users
* Create conda environment with required dependencies:

```shell script
conda env create --file conda_env.yml
conda activate test-env
```

#### Running the tool

* Download and setup tokenizer (approx. 300 MB of grammars for tree-sitter):

```shell script
python3 -m similar_repositories.setup_tokenizer
```

* List links to repositories in an input file (see [input_examples](input_examples) for examples).
* Run the tool. On the first run it will download several files with data (approx. 370 MB archived, 960 MB upacked):
```shell script
python3 -m similar_repositories.run -i input_examples/input.txt -o output
```

### Run the project from Docker (all platforms)

* Pull docker image
```shell script
docker pull egorbogomolov/similar-repositories:latest
```

* When running the docker container, bind `input`, `output`, and `data` directories in order to cache both auxiliary
 and output data, and inspect the output afterwards (e.g., check the vocabulary for analyzed projects). For Windows users,
 you should change the 

```shell script
docker run \
  --mount type=bind,source="$(pwd)"/input_examples,target=/input_examples/ \
  --mount type=bind,source="$(pwd)"/output,target=/output/ \
  --mount type=bind,source="$(pwd)"/data,target=/data \
  egorbogomolov/similar-repositories:latest -i input_examples/input.txt -o output/examples_output/
```

## Advanced options

1. `-f`, `--force` &ndash; if passed, all stages will be re-run, otherwise stored intermediate data will be used (if exists).
2. `-s`, `--min_stars` &ndash; searching for similar projects among those with at least `min_stars` stars. 
Available options are 10, 50, 100. 0 and 1 are coming soon. Default is 100.
3. `-k`, `--closest` &ndash; number of closest repositories to print. Default is 10.
4. `-b`, `--batches` &ndash; number of projects that are tokenized together and stored to one file. Default is 100.
5. `-m`, `--metric` &ndash; a method to compute project similarity, either `kl` or `cosine`. Default is `kl`.

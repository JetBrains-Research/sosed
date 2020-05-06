# Similar projects search

An embedding-based approach to detect simliar repositories on GitHub.

## How it works

We took a dataset of 9 million GitHub repositories from the [paper](https://arxiv.org/pdf/1704.00135.pdf) by Markovtsev et al.
It contains all the repositories on GitHub as of the end of 2016, excluding explicit and implicit forks (see paper for the details).

Then, we computed embeddings for tokens in code with [fasttext](https://github.com/facebookresearch/fastText) and clustered them
into 256 groups of similar tokens. The clusters represent topics at token level. For each repository we computed 
its numerical representation in the space token clusters. It is a 256-dimensional vector, where coordinate along the
dimension _K_ is the number of occurrences of tokens from cluster _K_ in the repository.

Finally, we measure similarity of two repositories as cosine distance between their vector representations. The intuition
behind this idea is that for repositories with similar topics the distribution of token-topics should also be similar.

## Getting started

* Create conda environment with required dependencies:

```shell script
conda env create --file conda_env.yml
conda activate test-env
```

* Download and setup tokenizer (approx. 300 MB of grammars for tree-sitter):

```shell script
python3 -m similar_repositories.setup_tokenizer
```

* List links to repositories in an input file (see [test_data](test_data) for examples).
* Run the tool. On the first run it will download several files with data (approx. 370 MB archived, 960 MB upacked):
```shell script
python3 -m similar_repositories.run -i test_data/input.txt -o output
```

## Advanced options

1. `-f`, `--force` &ndash; if passed, all stages will be re-run, otherwise stored intermediate data will be used (if exists).
2. `-s`, `--min_stars` &ndash; search for similar projects among those with at least `min_stars` stars. 
Available options are 10, 50, 100. 0 and 1 are coming soon. Default is 100.
3. `-k`, `--closest` &ndash; number of closest repositories to print. Default is 10.
4. `-b`, `--batches` &ndash; number of projects that are tokenized together and stored to one file. Default is 100.

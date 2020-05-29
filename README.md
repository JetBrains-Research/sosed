[![JetBrains Research](https://jb.gg/badges/research.svg)](https://research.jetbrains.org/groups/ml_methods)
[![Linux, MacOS, Docker](https://github.com/JetBrains-Research/similar-repositories/workflows/Linux,%20MacOS,%20Docker/badge.svg)](https://github.com/JetBrains-Research/similar-repositories/actions?query=workflow%3A%22Linux%2C+MacOS%2C+Docker%22)

# *Sosed*, similar projects search

An embedding-based approach to detect similar software projects.

## *Sosed*'s history

Initially, we created a novel approach to represent code as a part of source code topic modeling project (has not finished yet).
On the way, we found out that with the new representation, we can explicitly measure distance between arbitrary pieces of code (and even projects!),
and at project level it works reasonably well.
We decided to implement it as a stand-alone tool to verify feasibility of our approach and share it with the community.


## How *Sosed* works

### Topic-oriented code representation

Firstly, we took a [large corpus of sub-token sequences](https://data.world/vmarkovtsev/github-word-2-vec-120-k) and trained
their embeddings with [fasttext](https://github.com/facebookresearch/fastText).
Then, we clustered the embeddings with [spherical K-means](https://github.com/jasonlaska/spherecluster) to get 256 groups
of semantically similar tokens. Clusters reflect topics that occurred at sub-token level in a large corpus of source code. 

We represent code as a distribution of clusters among its sub-tokens. We hypothesize that fragments of code with similar 
distributions are also similar in a more broad sense.

### Searching for similar projects

We took a dataset of 9 million GitHub repositories from the [paper](https://arxiv.org/pdf/1704.00135.pdf) by Markovtsev et al.
It contains all the repositories on GitHub as of the end of 2016, excluding explicit and implicit forks (see paper for the details).

We computed the aforementioned representations for all the repositories. Cluster distribution can be seen as a 
256-dimensional vector, where coordinate along dimension _C_ is the probability of cluster _C_ appearing among project's
sub-tokens. We propose two ways to measure similarity of the distributions: either by using [KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
or cosine similarity. While the first option has clearer motivation from the mathematics perspective, our experience
with the tool shows that both options produce similar results.

It is worth noting, that in case of discrete distributions, both maximization of cosine similarity and minimization of 
KL-divergence can be reduced to maximization of two vectors' inner product. 

### Explaining the output

In order for *Sosed* not to act as a black-box, we manually labeled all 256 clusters with short descriptions of their topics.
If you pass `--explain` flag to *Sosed*, it will identify clusters that contributed the most to the similarity and 
print their descriptions.

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
conda activate similar-repositories-env
```

#### Running the tool

* Download and setup tokenizer (approx. 75 MB of grammars for tree-sitter):

```shell script
python3 -m similar_repositories.setup_tokenizer
```

* List links to GitHub repositories or paths to local projects in an input file (see [input_examples](input_examples) 
for examples).
* Run the tool. On the first run it will download several files with data (approx. 300 MB archived, 960 MB upacked):
```shell script
python3 -m similar_repositories.run -i input_examples/input.txt -o output/output_example
```

### Run the project from Docker

* Pull docker image
  ```shell script
  docker pull egorbogomolov/similar-repositories:latest
  ```

* Map `input`, `output`, and `data` directories from inside the container to the local filesystem, to cache both 
auxiliary and output data. This allows to inspect the output afterwards (e.g., check the vocabulary for analyzed 
projects) outside the container

  ```shell script
  docker run \
    -v "$(pwd)"/input_examples:/input_examples/ \
    -v "$(pwd)"/output:/output/ \
    -v "$(pwd)"/data:/data \
    egorbogomolov/similar-repositories:latest -i input_examples/input.txt -o output/examples_output/
  ```

## Known issues

* *Sosed* does not work from source on Windows, because some dependencies do not support Windows. Consider using 
[WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) or 
[Docker for Windows](https://docs.docker.com/docker-for-windows/).
* Clusters were labeled manually, and even though we tried our best, the range of topics is very broad, from "CPU fans"
to "Large Hadron Collider". Some descriptions may be imprecise, so feel free to open an issue in case you find 
any debatable descriptions!

## Advanced options

1. `-l', `--local` &ndash; if passed, the input file will be treated as a list of paths to local directories.
2. `-f`, `--force` &ndash; if passed, all stages will be re-run, otherwise stored intermediate data will be used.
3. `-s`, `--min_stars` &ndash; searching for similar projects among those with at least `min_stars` stars. 
Available options are 0, 1, 10, 50, 100. Default is 100. For 0+ and 1+ options, a large archive will be downloaded 
(0+ stars: 1 GB compressed, 9 GB decompressed; 1+ stars: 250 MB compressed, 2 GB decompressed).
4. `-k`, `--closest` &ndash; number of closest repositories to print. Default is 10.
5. `-b`, `--batches` &ndash; number of projects that are tokenized together and stored to one file. Default is 100.
6. `-m`, `--metric` &ndash; a method to compute project similarity, either `kl` or `cosine`. Default is `kl`.
7. `-e`, `--explain` &ndash; if passed, *Sosed* will explain the similarity.
8. `--lang` &ndash; language name. If passed, *Sosed* will output only the projects in given language.

### Example of usage with all the flags:
```shell script
python3 -m similar_repositories.run \
  -i input_examples/input.txt -o output_example \
  --local --force --min_stars 10 --closest 20 --batches 1000 --metric cosine --explain --lang Go
```

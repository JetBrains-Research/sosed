# Similar projects search

An embedding-based approach to detect simliar repositories on GitHub.

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
* Run the tool. On the first run it will download several files with data (approx. 400 MB):
```shell script
python3 -m similar_repositories.run -i test_data/input.txt -o output
```

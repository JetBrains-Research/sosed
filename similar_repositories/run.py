import os
import numpy as np

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .utils import mkdir
from .data_processing import ProcessedData, assign_clusters, compute_vectors
from tokenizer.topic_dynamics.run import main as run_tokenizer


def tokenize(input_file: str, tokenizer_output: str, batches: int, force: bool) -> None:
    """
    :param input_file: input file with a list of links to projects for analysis.
    :param tokenizer_output: directory to store data during tokenizing.
    :param batches: size of project batches that are saved to one file.
    :param force: if True, tokenizer will re-run (even if results have been stored previously).
    :return: None.
    """

    try:
        if not force and len(ProcessedData(Path(tokenizer_output)).indices()) > 0:
            print(f'Found tokenizer output in {tokenizer_output}.\n'
                  f'If you want to re-run tokenizer, pass --force flag.')
            return
    except ValueError:
        pass

    if not os.path.exists(input_file):
        raise ValueError(f'Input file {input_file} does not exist!')

    mkdir(tokenizer_output)
    tokenizer_args = Namespace(input=input_file, output=tokenizer_output, batches=batches)
    print(f'Running tokenizer on repos listed in {input_file}')
    run_tokenizer(tokenizer_args)


def vectorize(tokenizer_output: str, force: bool) -> None:
    """
    Compute numerical representations for repositories processed by tokenizer.
    :param tokenizer_output: directory where tokenizer stored extracted data about tokens in repositories.
    :param force: if True, vectorization will re-run (even if results have been stored previously).
    :return: None.
    """
    processed_data = ProcessedData(Path(tokenizer_output))
    if not force and processed_data.has_stored_repo_names() and processed_data.has_stored_repo_vectors():
        print(f'Found precomputed vectors in {tokenizer_output}.\n'
              f'If you wan to re-run vector computation, pass --force flag.')
        return

    all_repo_names = []
    all_vectors_list = []

    print(f'Found {len(processed_data.indices())} batches with tokenized data.')
    for ind in processed_data.indices():
        vocab = processed_data.load_tokens_vocab(ind)
        tokens_to_clusters = assign_clusters(vocab)
        docword = processed_data.load_docword(ind)
        repo_names, vectors = compute_vectors(docword, tokens_to_clusters)
        all_repo_names += repo_names
        all_vectors_list.append(vectors)

    all_vectors = np.concatenate(all_vectors_list)
    processed_data.store_repo_names(all_repo_names)
    processed_data.store_repo_vectors(all_vectors)


def analyze():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Full path to the input file with a list of links to GitHub.")
    parser.add_argument("-t", "--tokenizer_output", required=True,
                        help="Full path to the directory for storing tokenized data.")
    parser.add_argument("-b", "--batches", default=100,
                        help="The size of the batch of projects that are saved to one file.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="If passed, all stages will be re-run, otherwise stored data will be used.")
    args = parser.parse_args()

    tokenize(args.input, args.tokenizer_output, args.batches, args.force)
    vectorize(args.tokenizer_output, args.force)

import os

from pathlib import Path

DIM = 256

TOKENIZER_DIR = Path('tokenizer')
TOKENIZER_URL = 'https://github.com/areyde/topic-dynamics.git'
TOKENIZER_COMMIT = 'a8f8c0860d556eb98da32ef9bfec7ee1966fe98a'

DATA_DIR = Path('data')
CLUSTERS_FILE = 'clusters.npy'
TOKENS_FILE = 'tokens.txt'


def embedding_dim() -> int:
    return DIM


def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.isdir(path):
        raise ValueError(f'{path} is not a directory!')


def get_data_dir() -> Path:
    mkdir(DATA_DIR)
    return DATA_DIR


def get_clusters_file() -> Path:
    filepath = get_data_dir() / CLUSTERS_FILE

    if not os.path.exists(filepath):
        # TODO: download clusters file
        pass

    return filepath


def get_tokens_file() -> Path:
    filepath = get_data_dir() / TOKENS_FILE

    if not os.path.exists(filepath):
        # TODO: download tokens file
        pass

    return filepath

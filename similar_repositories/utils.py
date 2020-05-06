import os
import numpy as np

from pathlib import Path
from typing import List

DIM = 256

TOKENIZER_DIR = Path('tokenizer')
TOKENIZER_URL = 'https://github.com/areyde/topic-dynamics.git'
TOKENIZER_COMMIT = 'a8f8c0860d556eb98da32ef9bfec7ee1966fe98a'

DATA_DIR = Path('data')
CLUSTERS_FILE = 'clusters.npy'
TOKENS_FILE = 'tokens.txt'

VALID_STARS = [0, 1, 10, 50, 100]
REPO_NAMES_FILES = {stars: f'repo_names_{stars}.txt' for stars in VALID_STARS}
REPO_EMBED_FILES = {stars: f'repo_embed_{stars}.npy' for stars in VALID_STARS}


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

    if not filepath.exists():
        # TODO: download clusters file
        pass

    return filepath


def get_tokens_file() -> Path:
    filepath = get_data_dir() / TOKENS_FILE

    if not filepath.exists():
        # TODO: download tokens file
        pass

    return filepath


def is_valid_min_stars(min_stars: int) -> bool:
    return min_stars in VALID_STARS


def get_project_names(min_stars: int) -> List[str]:
    if not is_valid_min_stars(min_stars):
        raise ValueError(f'min_stars should be one of {VALID_STARS}, not {min_stars}')

    filepath = DATA_DIR / REPO_NAMES_FILES[min_stars]

    if not filepath.exists():
        # TODO: download names file
        pass

    return [line.strip() for line in filepath.open('r')]


def get_project_vectors(min_stars: int) -> np.ndarray:
    if not is_valid_min_stars(min_stars):
        raise ValueError(f'min_stars should be one of {VALID_STARS}, not {min_stars}')

    filepath = DATA_DIR / REPO_EMBED_FILES[min_stars]

    if not filepath.exists():
        # TODO: download embeddings file
        pass

    return np.load(filepath, allow_pickle=True)

import os
import numpy as np

from pathlib import Path
from typing import List

DIM = 256

TOKENIZER_DIR = Path('tokenizer')
TOKENIZER_URL = 'https://github.com/JetBrains-Research/identifiers-extractor.git'
TOKENIZER_VERSION = 'v1.0.0'

DATA_LINK = 'https://drive.google.com/uc?id=1CTO24nZtMyHVQ43mhljfH2qN_QWNIZGo'
DATA_ARCHIVE = 'data.tar.gz'
DATA_DIR = Path('data')
CLUSTERS_FILE = 'clusters.npy'
TOKENS_FILE = 'tokens.txt'

VALID_STARS = [10, 50, 100]
REPO_NAMES_FILES = {stars: f'repo_names_{stars}.txt' for stars in VALID_STARS}
REPO_EMBED_FILES = {stars: f'repo_embed_{stars}.npy' for stars in VALID_STARS}


def embedding_dim() -> int:
    return DIM


def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.isdir(path):
        raise ValueError(f'{path} is not a directory!')


def download_data() -> None:
    os.system(f'gdown {DATA_LINK}')
    mkdir(DATA_DIR)
    os.system(f'tar -xzvf {DATA_ARCHIVE} -C {DATA_DIR} --strip-components 1')
    os.remove(DATA_ARCHIVE)


def get_data_dir() -> Path:
    mkdir(DATA_DIR)
    return DATA_DIR


def get_clusters_file() -> Path:
    filepath = get_data_dir() / CLUSTERS_FILE

    if not filepath.exists():
        download_data()

    return filepath


def get_tokens_file() -> Path:
    filepath = get_data_dir() / TOKENS_FILE

    if not filepath.exists():
        download_data()

    return filepath


def is_valid_min_stars(min_stars: int) -> bool:
    return min_stars in VALID_STARS


def get_project_names(min_stars: int) -> List[str]:
    if not is_valid_min_stars(min_stars):
        raise ValueError(f'min_stars should be one of {VALID_STARS}, not {min_stars}')

    filepath = DATA_DIR / REPO_NAMES_FILES[min_stars]

    if not filepath.exists():
        download_data()

    return [line.strip() for line in filepath.open('r')]


def get_project_vectors(min_stars: int) -> np.ndarray:
    if not is_valid_min_stars(min_stars):
        raise ValueError(f'min_stars should be one of {VALID_STARS}, not {min_stars}')

    filepath = DATA_DIR / REPO_EMBED_FILES[min_stars]

    if not filepath.exists():
        download_data()

    return np.load(filepath, allow_pickle=True)

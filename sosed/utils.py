import os
import pickle
import numpy as np

from pathlib import Path
from typing import List, Tuple

DIM = 256
# Number of tokens to keep in repository stats
TOP_TOKENS = 5

TOKENIZER_DIR = Path('tokenizer')
TOKENIZER_URL = 'https://github.com/JetBrains-Research/identifiers-extractor.git'
TOKENIZER_VERSION = 'v1.2.1'

DATA_LINK = 'https://s3-eu-west-1.amazonaws.com/resources.ml.labs.aws.intellij.net/sosed/{}'
DATA_TC_ARCHIVE = 'data_tc.tar.xz'
DATA_STARS_ARCHIVE = 'data_stars_{}.tar.xz'
DATA_DIR = Path('data')
CLUSTERS_FILE = 'clusters.npy'
CLUSTERS_INFO_FILE = 'clusters_info.pkl'
TOKENS_FILE = 'tokens.txt'

VALID_STARS = [10, 50, 100]
REPO_NAMES_FILES = {stars: f'repo_names_{stars}.txt' for stars in VALID_STARS}
REPO_EMBED_FILES = {stars: f'repo_embed_{stars}.npy' for stars in VALID_STARS}
REPO_LANGUAGES_FILES = {stars: f'repo_languages_{stars}.pkl' for stars in VALID_STARS}


def embedding_dim() -> int:
    return DIM


def stats_top_tokens() -> int:
    return TOP_TOKENS


def mkdir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)

    if not os.path.isdir(path):
        raise ValueError(f'{path} is not a directory!')


def download_tc_data() -> None:
    mkdir(DATA_DIR)

    os.system(f'wget {DATA_LINK.format(DATA_TC_ARCHIVE)} || curl -O {DATA_LINK.format(DATA_TC_ARCHIVE)}')
    os.system(f'tar -xvf {DATA_TC_ARCHIVE} -C {DATA_DIR}')
    os.remove(DATA_TC_ARCHIVE)


def download_data(min_stars: int) -> None:
    mkdir(DATA_DIR)

    data_stars = DATA_STARS_ARCHIVE.format(min_stars)
    os.system(f'wget {DATA_LINK.format(data_stars)} || curl -O {DATA_LINK.format(DATA_TC_ARCHIVE)}')
    os.system(f'tar -xvf {data_stars} -C {DATA_DIR}')
    os.remove(data_stars)


def get_data_dir() -> Path:
    mkdir(DATA_DIR)
    return DATA_DIR


def get_clusters_file() -> Path:
    filepath = get_data_dir() / CLUSTERS_FILE

    if not filepath.exists():
        download_tc_data()

    return filepath


def get_tokens_file() -> Path:
    filepath = get_data_dir() / TOKENS_FILE

    if not filepath.exists():
        download_tc_data()

    return filepath


def is_valid_min_stars(min_stars: int) -> bool:
    return min_stars in VALID_STARS


def get_project_names(min_stars: int) -> List[str]:
    if not is_valid_min_stars(min_stars):
        raise ValueError(f'min_stars should be one of {VALID_STARS}, not {min_stars}')

    filepath = DATA_DIR / REPO_NAMES_FILES[min_stars]

    if not filepath.exists():
        download_data(min_stars)

    return [line.strip() for line in filepath.open('r')]


def get_project_vectors(min_stars: int) -> np.ndarray:
    if not is_valid_min_stars(min_stars):
        raise ValueError(f'min_stars should be one of {VALID_STARS}, not {min_stars}')

    filepath = DATA_DIR / REPO_EMBED_FILES[min_stars]

    if not filepath.exists():
        download_data(min_stars)

    return np.load(filepath, allow_pickle=True)


def get_project_languages(min_stars: int) -> List[str]:
    if not is_valid_min_stars(min_stars):
        raise ValueError(f'min_stars should be one of {VALID_STARS}, not {min_stars}')

    filepath = DATA_DIR / REPO_LANGUAGES_FILES[min_stars]

    if not filepath.exists():
        download_data(min_stars)

    return pickle.load(open(filepath, 'rb'))


def get_clusters_info() -> List[Tuple[str, List[str]]]:
    filepath = DATA_DIR / CLUSTERS_INFO_FILE

    return pickle.load(filepath.open('rb'))

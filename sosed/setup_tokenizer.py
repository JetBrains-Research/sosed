import os

from .utils import *
from tokenizer.identifiers_extractor.parsers.utils import main as init_libraries


def setup_tokenizer() -> None:
    """
    Clone GitHub repository with tokenizer and setup it.
    :return: None.
    """
    os.system(f'git clone --branch={TOKENIZER_VERSION} {TOKENIZER_URL} {TOKENIZER_DIR}')
    os.chdir(TOKENIZER_DIR)
    os.system('git submodule update --init --recursive --depth 1')
    os.chdir('..')
    init_libraries()


if __name__ == '__main__':
    setup_tokenizer()

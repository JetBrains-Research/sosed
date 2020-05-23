import os

from .utils import *


def setup_tokenizer() -> None:
    """
    Clone GitHub repository with tokenizer and setup it.
    :return: None.
    """
    os.system(f'git clone --recursive --shallow-submodules {TOKENIZER_URL} {TOKENIZER_DIR}')
    os.chdir(TOKENIZER_DIR)
    os.system(f'git checkout {TOKENIZER_VERSION} --quiet')
    os.chdir('..')


if __name__ == '__main__':
    setup_tokenizer()

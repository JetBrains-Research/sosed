import os

from .utils import *


def setup_tokenizer() -> None:
    """
    Clone GitHub repository with tokenizer and setup it.
    :return: None.
    """
    # TODO: Compare speed with different cloning options
    os.system(f'git clone --recurse-submodules --branch={TOKENIZER_VERSION} {TOKENIZER_URL} {TOKENIZER_DIR}')


if __name__ == '__main__':
    setup_tokenizer()

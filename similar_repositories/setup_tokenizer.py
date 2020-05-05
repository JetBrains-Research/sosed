import os

from .utils import *


def setup_tokenizer() -> None:
    os.system(f'git clone --recurse-submodules {TOKENIZER_URL} {TOKENIZER_DIR}')
    os.chdir(TOKENIZER_DIR)
    os.system(f'git checkout {TOKENIZER_COMMIT}')
    os.chdir('..')


if __name__ == '__main__':
    setup_tokenizer()

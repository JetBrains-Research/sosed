import os
import numpy as np
import faiss

from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from tqdm import tqdm

from .utils import get_clusters_file, get_tokens_file, embedding_dim

__all__ = [
    'ProcessedData',
    'assign_clusters', 'compute_vectors', 'normalize_vectors', 'build_similarity_index', 'get_top_supertokens'
]

class ProcessedData:
    """
    Wrapper to ease work with output of tokenizer and extracted vector representations.
    """
    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._docword_files = {
            self._docword_index(f): folder / f
            for f in os.listdir(folder)
            if f.startswith('docword')
        }
        self._vocab_files = {
            self._vocab_index(f): folder / f
            for f in os.listdir(folder)
            if f.startswith('vocab')
        }

        if set(self._docword_files.keys()) != set(self._vocab_files.keys()):
            raise ValueError(f'Incorrect output by tokenizer. Indices of docword files do not match vocab files.\n'
                             f'{self._docword_files.keys()} | {self._vocab_files.keys()}')

        self._indices = list(self._docword_files.keys())
        self._tokens_vocab = {ind: None for ind in self._indices}
        self._docword = {ind: None for ind in self._indices}

        self._repo_names_file = folder / 'repo_names.txt'
        self._repo_vectors_file = folder / 'repo_vectors.npy'
        self._repo_names = None
        self._repo_vectors = None

    @staticmethod
    def _docword_index(filename: str) -> int:
        return int(filename[len('docword'):][:-len('.txt')])

    @staticmethod
    def _vocab_index(filename: str) -> int:
        return int(filename[len('vocab'):][:-len('.txt')])

    def indices(self) -> List[int]:
        """
        :return: batch indices found in tokenizer output.
        """
        return self._indices

    def folder(self) -> Path:
        return self._folder

    def load_tokens_vocab(self, ind: int) -> Dict[str, int]:
        """
        :param ind: index of vocab file.
        :return: mapping from tokens to their indices.
        """
        if self._tokens_vocab[ind] is None:
            vocab = {}

            with self._vocab_files[ind].open('r') as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        index, token = line.split(';')
                        vocab[token] = int(index)

            self._tokens_vocab[ind] = vocab

        return self._tokens_vocab[ind]

    def load_docword(self, ind: int) -> Dict[str, Counter]:
        """
        :param ind: index of docword file.
        :return: mapping from repository names to counts of tokens in them.
        """
        if self._docword[ind] is None:
            docword = {}

            with self._docword_files[ind].open('r') as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        repo_name, rest = line.split(';')
                        token_counter = Counter()
                        for token_count in rest.split(','):
                            token_ind, count = token_count.split(':')
                            token_counter[int(token_ind)] = int(count)
                        docword[repo_name] = token_counter

            self._docword[ind] = docword

        return self._docword[ind]

    def store_repo_names(self, repo_names: List[str]) -> None:
        self._repo_names = repo_names
        with self._repo_names_file.open('w') as repo_names_file:
            for name in repo_names:
                repo_names_file.write(f'{name}\n')

    def store_repo_vectors(self, repo_vectors: np.ndarray) -> None:
        self._repo_vectors = repo_vectors
        np.save(str(self._repo_vectors_file)[:-len('.npy')], repo_vectors, allow_pickle=True)

    def has_stored_repo_names(self) -> bool:
        return self._repo_names_file.exists()

    def has_stored_repo_vectors(self) -> bool:
        return self._repo_vectors_file.exists()

    def load_repo_names(self) -> List[str]:
        if self._repo_names is None:
            self._repo_names = [line.strip() for line in self._repo_names_file.open('r')]
        return self._repo_names

    def load_repo_vectors(self) -> np.ndarray:
        if self._repo_vectors is None:
            self._repo_vectors = np.load(self._repo_vectors_file, allow_pickle=True)
        return self._repo_vectors


def assign_clusters(tokens_vocab: Dict[str, int]) -> Dict[int, int]:
    """
    Build a new dictionary that maps token indices to cluster indices.
    :param tokens_vocab: mapping from tokens to their indices.
    :return: mapping from token indices to cluster indices, None if token was not found.
    """
    print(f'Assigning clusters to tokens from vocab file.')
    tokens_to_clusters = {token_ind: None for token_ind in tokens_vocab.values()}

    clusters = np.load(get_clusters_file(), allow_pickle=True)
    with get_tokens_file().open('r') as tokens_file:
        for token, cluster in tqdm(zip(tokens_file, clusters), total=len(clusters)):
            token = token.strip()
            if token in tokens_vocab:
                tokens_to_clusters[tokens_vocab[token]] = int(cluster)

    return tokens_to_clusters


def compute_vectors(docword: Dict[str, Counter], tokens_to_clusters: Dict[int, int]) -> Tuple[List[str], np.ndarray]:
    """
    :param docword: mapping from repo names to token counters.
    :param tokens_to_clusters: mapping from token indices to cluster indices, None if token was not found.
    :return: tuple of 1) list of repository names, 2) numpy matrix (N_repos \times N_clusters), each row is a vector
    representation of a repository.
    """
    print(f'Computing vectors for {len(docword)} repositories.')
    repo_names = []
    vectors = np.zeros((len(docword), embedding_dim()), dtype=np.float32)

    for i, (repo_name, token_counts) in enumerate(docword.items()):
        repo_names.append(repo_name)
        for token, count in token_counts.items():
            cluster = tokens_to_clusters[token]
            if cluster is not None:
                vectors[i][cluster] += count

    return repo_names, vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def build_similarity_index(embedding: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embedding_dim())
    index.add(embedding)
    return index


def get_top_supertokens(repo_vector: np.ndarray, index: faiss.IndexFlatIP, ind: int, k=10) -> List[Tuple[int, np.float]]:
    dot_product = index.reconstruct(ind) * repo_vector
    idx = reversed(np.argsort(dot_product)[-k:])
    return [(dim, dot_product[dim]) for dim in idx]

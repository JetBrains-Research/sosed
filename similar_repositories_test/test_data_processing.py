import unittest
import os
import numpy as np

from pathlib import Path
from collections import Counter
from unittest.mock import patch

from similar_repositories.data_processing import *


class ProcessedDataTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.actual_index = [0, 1]
        cls.actual_docword = {
            0: {
                'project1': Counter({1: 1, 2: 2, 3: 3, 5: 4}),
                'project2': Counter({1: 2, 3: 10, 4: 100})
            },
            1: {
                'project5': Counter({10: 2, 30: 10, 40: 100}),
                'project4': Counter({1: 2, 3: 4, 5: 6, 7: 8, 9: 10}),
                'project3': Counter({10: 10, 20: 220, 33: 333, 5: 1}),
            }
        }
        cls.actual_vocab = {
            0: {
                'a': 1,
                'bb': 2,
                'ccc': 3,
                'dddd': 4,
                'eeeee': 5
            },
            1: {
                'on': 1,
                'going': 3,
                'is': 5,
                'weird': 7,
                'something': 9,
                'thirtythree': 33,
                'a': 10,
                'bb': 20,
                'ccc': 30,
                'dddd': 40,
                'eeeee': 50
            }
        }

        cls.folder = Path('test_data', 'test_output')
        cls.processed_data = ProcessedData(cls.folder)

    def test_indices(self):
        self.assertEqual(self.actual_index, self.processed_data.indices())

    def test_docword(self):
        self.assertEqual(self.actual_docword[0], self.processed_data.load_docword(0))
        self.assertEqual(self.actual_docword[1], self.processed_data.load_docword(1))

    def test_vocab(self):
        self.assertEqual(self.actual_vocab[0], self.processed_data.load_tokens_vocab(0))
        self.assertEqual(self.actual_vocab[1], self.processed_data.load_tokens_vocab(1))

    def test_repo_names(self):
        self.assertFalse(self.processed_data.has_stored_repo_names())
        repo_names = ['project1', 'project2', 'project5', 'project4', 'project3']
        self.processed_data.store_repo_names(repo_names)
        self.assertTrue(self.processed_data.has_stored_repo_names())
        self.assertEqual(repo_names, self.processed_data.load_repo_names())

    def test_repo_vectors(self):
        self.assertFalse(self.processed_data.has_stored_repo_vectors())
        repo_vectors = np.random.randn(10, 20)
        self.processed_data.store_repo_vectors(repo_vectors)
        self.assertTrue(self.processed_data.has_stored_repo_vectors())
        self.assertTrue(np.all(repo_vectors == self.processed_data.load_repo_vectors()))

    @classmethod
    def tearDownClass(cls):
        names_file = cls.folder / 'repo_names.txt'
        if names_file.exists():
            os.remove(names_file)

        vectors_file = cls.folder / 'repo_vectors.npy'
        if vectors_file.exists():
            os.remove(vectors_file)


class DataProcessingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folder = Path('test_data', 'test_data')
        cls.tokens_file = cls.folder / 'tokens.txt'
        cls.clusters_file = cls.folder / 'clusters.npy'
        cls.real_tokens = ['we', 'are', 'the', 'champions', 'my', 'friends', 'and', 'keep', 'on', 'fighting', 'till', 'end']
        cls.real_clusters = np.arange(len(cls.real_tokens))
        cls.short_tokens = cls.real_tokens[::2]
        cls.short_clusters = cls.real_clusters[::2]
        np.save(cls.folder / 'clusters', cls.short_clusters, allow_pickle=True)

    @patch('similar_repositories.data_processing.get_clusters_file')
    @patch('similar_repositories.data_processing.get_tokens_file')
    def test_assign_clusters(self, mock_get_tokens_file, mock_get_clusters_file):
        mock_get_clusters_file.return_value = self.clusters_file
        mock_get_tokens_file.return_value = self.tokens_file

        tokens_vocab = {token: i for i, token in enumerate(self.real_tokens)}
        proper_assignment = {ind: None for ind in tokens_vocab.values()}
        proper_assignment.update({
            tokens_vocab[token]: cluster
            for token, cluster in zip(self.short_tokens, self.short_clusters) if token in tokens_vocab
        })
        self.assertEqual(proper_assignment, assign_clusters(tokens_vocab))

    @patch('similar_repositories.data_processing.embedding_dim')
    def test_compute_vectors(self, mock_embedding_dim):
        n_projects = 3
        dim = 8

        mock_embedding_dim.return_value = dim

        actual_repo_names = [f'project_{i}' for i in range(1, n_projects + 1)]
        tokens_to_clusters = {i: i % dim for i in range(dim * dim)}
        docword = {
            project: Counter({token: i + 1 for token in tokens_to_clusters})
            for i, project in enumerate(actual_repo_names)
        }
        actual_vectors = np.array([[i * dim for _ in range(dim)] for i in range(1, n_projects + 1)], dtype=np.float32)

        repo_names, vectors = compute_vectors(docword, tokens_to_clusters)

        self.assertEqual(actual_repo_names, repo_names)
        self.assertEqual((n_projects, dim), vectors.shape)
        self.assertTrue(np.all(actual_vectors == vectors))

    def test_normalize_vectors(self):
        n_projects = 3
        dim = 8

        vectors = np.random.randn(n_projects, dim)
        normalized_vectors = normalize_vectors(vectors)

        self.assertEqual((n_projects, dim), normalized_vectors.shape)
        for vec, norm_vec in zip(vectors, normalized_vectors):
            actual_norm_vec = vec / np.linalg.norm(vec)
            for i in range(dim):
                self.assertAlmostEqual(actual_norm_vec[i], norm_vec[i])

    @patch('similar_repositories.data_processing.embedding_dim')
    def test_similarity_index(self, mock_embedding_dim):
        n_projects = 10
        dim = 16

        mock_embedding_dim.return_value = dim

        embedding = np.random.random((n_projects, dim)).astype('float32')
        embedding[:, 0] += np.arange(n_projects) / 1000.
        embedding = normalize_vectors(embedding)

        index = build_similarity_index(embedding)
        dist, idx = index.search(embedding, 3)

        for i, inds in enumerate(idx):
            self.assertEqual(i, inds[0])

    @classmethod
    def tearDownClass(cls):
        clusters_file = cls.clusters_file
        if clusters_file.exists():
            os.remove(clusters_file)


if __name__ == '__main__':
    unittest.main()

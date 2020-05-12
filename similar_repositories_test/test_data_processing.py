import unittest
import os
import numpy as np

from pathlib import Path
from collections import Counter

from similar_repositories.data_processing import ProcessedData


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


if __name__ == '__main__':
    unittest.main()

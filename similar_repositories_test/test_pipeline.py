import unittest
import os
import shutil

from pathlib import Path


class PipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_file = Path('input_examples', 'input.txt')
        cls.output_dir = Path('test_data', 'tokenizer_output')
        cls.output_file = Path('test_data', 'output.txt')
        cls.repo_names = [line.strip() for line in cls.input_file.open('r').readlines() if line.strip() != '']

    def test_pipeline(self):
        min_stars = 50
        k = 9

        os.system(
            f'python3 -m similar_repositories.run '
            f'-i {self.input_file} -o {self.output_dir} -k {k} -s {min_stars} '
            f'> {self.output_file}'
        )

        encountered_repos = []
        similarities = []
        block_header = 'Top picks for '

        output = [line.strip() for line in self.output_file.open('r').readlines()]
        for line in output:
            if line.startswith(block_header):
                encountered_repos.append(line[len(block_header):])
                similarities.append([])
            elif line.startswith('https'):
                similarities[-1].append(float(line.split(' | ')[1]))

        self.assertEqual(self.repo_names, encountered_repos)

        for sims in similarities:
            self.assertEqual(k, len(sims))
            for similarity in sims:
                self.assertTrue(similarity > 0.7)
            for similarity1, similarity2 in zip(sims[:-1], sims[1:]):
                self.assertTrue(similarity1 >= similarity2)

    @classmethod
    def tearDownClass(cls):
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
        if cls.output_file.exists():
            os.remove(cls.output_file)


if __name__ == '__main__':
    unittest.main()

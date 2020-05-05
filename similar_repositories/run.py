import os

from argparse import ArgumentParser, Namespace

from tokenizer.topic_dynamics.run import main as run_tokenizer


def tokenize(input_file: str, tokenizer_output: str, batches: int) -> None:
    """
    :param input_file: input file with a list of links to projects for analysis.
    :param tokenizer_output: directory to store data during tokenizing.
    :param batches: size of project batches that are saved to one file.
    :return: None.
    """

    if not os.path.exists(input_file):
        raise ValueError(f'Input file {input_file} does not exist!')

    if not os.path.exists(tokenizer_output):
        os.mkdir(tokenizer_output)

    if not os.path.isdir(tokenizer_output):
        raise ValueError(f'{tokenizer_output} is not a directory!')

    tokenizer_args = Namespace(input=input_file, output=tokenizer_output, batches=batches)
    run_tokenizer(tokenizer_args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Full path to the input file with a list of links to GitHub.")
    parser.add_argument("-t", "--tokenizer_output", required=True,
                        help="Full path to the directory for storing tokenized data.")
    parser.add_argument("-b", "--batches", default=100,
                        help="The size of the batch of projects that are saved to one file.")
    args = parser.parse_args()

    tokenize(args.input, args.tokenizer_output, args.batches)

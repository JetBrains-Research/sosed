from argparse import ArgumentParser
from pathlib import Path

from sosed.data_processing import ProcessedData
from sosed.run import tokenize, vectorize, analyze_topics

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input file with a list of links to GitHub.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the directory for storing extracted data.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="If passed, switches the tokenization into the local mode, where "
                             "the input list must contain paths to local directories.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="If passed, all stages will be re-run, otherwise stored data will be used.")
    args = parser.parse_args()
    tokenize(args.input, args.output, 1, args.local, args.force, True)
    processed_data = ProcessedData(Path(args.output))
    vectorize(processed_data, args.force, True)
    analyze_topics(processed_data, args.output)

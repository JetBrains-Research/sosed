import os
import numpy as np

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .utils import mkdir, get_project_names, get_project_vectors, get_clusters_info
from .data_processing import *
from tokenizer.identifiers_extractor.run import main as run_tokenizer


def tokenize(input_file: str, output_dir: str, batches: int, local: str, force: bool) -> None:
    """
    :param input_file: input file with a list of links to projects for analysis.
    :param output_dir: directory to store data during tokenizing.
    :param batches: size of project batches that are saved to one file.
    :param local: if True, input file should contain paths to local projects.
    :param force: if True, tokenizer will re-run (even if results have been stored previously).
    :return: None.
    """

    mkdir(output_dir)
    try:
        if not force and len(ProcessedData(Path(output_dir)).indices()) > 0:
            print(f'Found tokenizer output in {output_dir}.\n'
                  f'If you want to re-run tokenizer, pass --force flag.')
            return
    except ValueError:
        pass

    if not os.path.exists(input_file):
        raise ValueError(f'Input file {input_file} does not exist!')

    tokenizer_args = Namespace(input=input_file, output=output_dir, batches=batches, local=local)
    print(f'Running tokenizer on repos listed in {input_file}')
    run_tokenizer(tokenizer_args)


def vectorize(processed_data: ProcessedData, force: bool) -> None:
    """
    Compute numerical representations for repositories processed by tokenizer.
    :param processed_data: wrapper for directory where tokenizer stored extracted data about tokens in repositories.
    :param force: if True, vectorization will re-run (even if results have been stored previously).
    :return: None.
    """
    if not force and processed_data.has_stored_repo_names() \
            and processed_data.has_stored_repo_vectors() \
            and processed_data.has_stored_repo_stats():
        print(f'Found precomputed vectors in {processed_data.folder()}.\n'
              f'If you wan to re-run vector computation, pass --force flag.')
        return

    all_repo_names = []
    all_vectors_list = []
    all_repo_stats = {}

    print(f'Found {len(processed_data.indices())} batches with tokenized data.')
    for ind in processed_data.indices():
        vocab = processed_data.load_tokens_vocab(ind)
        tokens_to_clusters = assign_clusters(vocab)
        docword = processed_data.load_docword(ind)
        repo_names, vectors = compute_vectors(docword, tokens_to_clusters)
        repo_stats = compute_repo_stats(docword, tokens_to_clusters, vocab)
        all_repo_names += repo_names
        all_vectors_list.append(vectors)
        all_repo_stats.update(repo_stats)

    all_vectors = np.concatenate(all_vectors_list)
    processed_data.store_repo_names(all_repo_names)
    processed_data.store_repo_vectors(all_vectors)
    processed_data.store_repo_stats(all_repo_stats)


def analyze(
        processed_data: ProcessedData, min_stars: int, closest: int, explain: bool, metric: str, language: str
) -> None:
    """
    Find similar projects for repositories based on their numerical representations.
    :param processed_data: wrapper for directory with numerical representations of projects.
    :param min_stars: threshold for the number of stars for reference repositories.
    :param closest: number of similar projects to output.
    :param explain: whether to output explanation of project similarity.
    :param metric: either `kl` or `cosine`, a way to compute project similarity. `kl` sorts projects based on the
    KL-divergence of their topic distributions. `cosine` sorts projects based on the cosine similarity of their
    representaitons.
    :param language: if not None, analysis uses only projects with this language.
    :return:
    """
    project_vectors = get_project_vectors(min_stars)
    project_names = get_project_names(min_stars)
    if language is not None:
        project_vectors, project_names = filter_by_language(project_vectors, project_names, language, min_stars)
        if len(project_vectors) == 0:
            raise ValueError(f"There are no projects for language {language}")

    if metric == 'kl':
        project_embed = kl_vectors(project_vectors)
        repo_vectors = probability_vectors(smooth_vectors(processed_data.load_repo_vectors()))
    elif metric == 'cosine':
        project_embed = normalize_vectors(project_vectors)
        repo_vectors = normalize_vectors(processed_data.load_repo_vectors())
    else:
        raise ValueError('Metric should be either "kl" or "cosine"')

    repo_names = processed_data.load_repo_names()
    repo_stats = processed_data.load_repo_stats()
    clusters_info = get_clusters_info()

    index = build_similarity_index(project_embed)

    distances, indices = index.search(repo_vectors, closest)

    for repo_name, repo_vector, dist_vector, idx in zip(repo_names, repo_vectors, distances, indices):
        print()
        print('-----------------------')
        print(f'Query project: {repo_name}')

        stats = repo_stats[repo_name]

        if explain:
            top_tokens = stats['top_tokens']
            top_tokens_string = ', '.join([f'{token} ({count})' for token, count in top_tokens])
            print()
            print(f'Most frequent sub-tokens: {top_tokens_string}')
            print(f'Top sub-token topics:')
            top_topics = np.argsort(repo_vector)[-5:][::-1]
            for dim in top_topics:
                print(
                    f'weight = {repo_vector[dim]:.2f} | {clusters_info[dim][0]:50s} | {", ".join([f"{token} ({count})" for token, count in stats["top_by_cluster"][dim]])}'
                )
            print()

        for ind, dist in zip(idx, dist_vector):
            print(f'https://github.com/{project_names[ind]} | similarity = {dist:.4f}')

            if explain:
                top_supertokens = get_top_supertokens(repo_vector, index, int(ind), metric)
                print()
                print('Intersecting topics:')
                print('\n'.join([
                    # f'{dim:3d} | intersection = {product / dist:.2f} | {", ".join([f"{token}" for token, count in stats["top_by_cluster"][dim]])}'
                    f'intersection = {product / dist:.2f} | {clusters_info[dim][0]:50s} | {", ".join([f"{token}" for token in clusters_info[dim][1]])}'
                    for dim, product in top_supertokens
                ]))
                print()

        print('-----------------------')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input file with a list of links to GitHub.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the directory for storing extracted data.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="If passed, switches the tokenization into the local mode, where "
                             "the input list must contain paths to local directories.")
    parser.add_argument("-b", "--batches", default=100, type=int,
                        help="The size of the batch of projects that are saved to one file.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="If passed, all stages will be re-run, otherwise stored data will be used.")
    parser.add_argument("-s", "--min_stars", default=100, type=int,
                        help="Find similar projects among projects with at least min_stars stars. "
                             "Valid options are 0, 1, 10, 50, 100.")
    parser.add_argument("-k", "--closest", default=10, type=int,
                        help="Number of closest repositories to find.")
    parser.add_argument("-e", "--explain", action="store_true",
                        help="If passed, the output will contain top super-tokens matched with each repository.")
    parser.add_argument("-m", "--metric", default="kl",
                        help="Metric to compute project similarity. Options are 'kl' (default) and 'cosine'")
    parser.add_argument("--lang", default=None, type=str,
                        help="If passed, specifies the language of reference projects. "
                             "Notice, that language data was extracted with GHTorrent and for some projects language "
                             "information is missing.")
    args = parser.parse_args()

    tokenize(args.input, args.output, args.batches, args.local, args.force)
    processed_data = ProcessedData(Path(args.output))
    vectorize(processed_data, args.force)
    analyze(processed_data, args.min_stars, args.closest, args.explain, args.metric, args.lang)

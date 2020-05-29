FROM python:3.8-slim

# g++ required by tree-sitter
RUN apt-get update
RUN apt-get install -y --no-install-recommends g++ git wget xz-utils && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install cython && pip install -r requirements.txt

COPY data/clusters_info.pkl data/
COPY similar_repositories/ similar_repositories/
RUN python -m similar_repositories.setup_tokenizer
RUN python -m tokenizer.identifiers_extractor.parsers
RUN python -m tokenizer.identifiers_extractor.language_recognition

ENTRYPOINT ["python", "-m", "similar_repositories.run"]
CMD ["--help"]

FROM python:3.8-slim

# g++ required by tree-sitter
RUN apt-get update
RUN apt-get install -y --no-install-recommends g++ git wget xz-utils && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install cython && pip install -r requirements.txt

COPY data/clusters_info.pkl data/
COPY sosed/ sosed/
RUN python -m sosed.setup_tokenizer
RUN python -m tokenizer.identifiers_extractor.parsers
RUN python -m tokenizer.identifiers_extractor.language_recognition

ENTRYPOINT ["python", "-m", "sosed.run"]
CMD ["--help"]

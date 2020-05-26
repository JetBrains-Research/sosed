FROM python:3.8-slim

# gcc required by tree-sitter
RUN apt-get update
RUN apt-get install -y --no-install-recommends g++

COPY requirements.txt .
RUN pip install cython
RUN pip install -r requirements.txt

COPY similar_repositories/ similar_repositories/
RUN apt-get install -y git
RUN python -m similar_repositories.setup_tokenizer
RUN python -m tokenizer.identifiers_extractor.parsers
RUN python -m tokenizer.identifiers_extractor.language_recognition

ENTRYPOINT ["python", "-m", "similar_repositories.run"]
CMD ["--help"]

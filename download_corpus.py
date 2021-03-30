import tarfile
import ssl
import shutil

from urllib.request import urlopen
from urllib.parse import urlparse
from pathlib import Path
from itertools import chain

import requests
import spacy

from tqdm import tqdm
from halo import Halo


SPACY_MODELS = {
    "fra": spacy.load("fr_core_news_sm"),
    "eng": spacy.load("en_core_web_sm"),
}


def lazzy_download(uri, filename=None):
    if filename is None:
        path = Path(urlparse(uri).path)
        filename = Path(path.name)

    context = ssl._create_unverified_context()
    if not filename.is_file():
        resp = requests.get(uri, stream=True)
        total = int(resp.headers.get("content-length", 0))

        with open(filename, "wb") as file, tqdm(
            total=total, unit="iB", unit_scale=True, unit_divisor=1024
        ) as bar:
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    return filename


def create_directory(directory_name):
    directory_name = Path(directory_name)
    directory_name.mkdir(parents=True, exist_ok=True)
    return directory_name


def extract_corpus(archive, original_filename, target_directory):
    print(f"Extracting {original_filename}")

    target_filename = target_directory / original_filename
    if target_filename.is_file():
        print(f"skip, {target_filename} already exists")
        return target_filename

    with Halo(text=f"Extracting {original_filename}", spinner="dots"):
        with tarfile.open(archive, "r") as ifile:
            original_filename = ifile.getmember(original_filename)
            ifile.extractall(target_directory, [original_filename])

    return target_filename


def tokenize_with_spacy(input_files, output_fn, language):

    print("Tokenize data with spacy")
    nlp = SPACY_MODELS[language]

    total_size = sum(1 for _ in chain.from_iterable([open(fn) for fn in input_files]))

    with open(output_fn, "w") as ofile:
        for line in tqdm(
            chain.from_iterable(open(fn) for fn in input_files), total=total_size
        ):
            tokenized_sentence = [
                token.text.strip() for token in nlp(line, disable=["parser", "ner"])
            ]
            ofile.write(" ".join(tokenized_sentence))
            ofile.write("\n")

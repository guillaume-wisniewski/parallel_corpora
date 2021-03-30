import tarfile
import ssl
import shutil

from urllib.request import urlopen
from urllib.parse import urlparse
from pathlib import Path
from itertools import chain

import spacy

from tqdm import tqdm
from halo import Halo


SPACY_MODELS = {"fra": spacy.load("fr_core_news_sm"),
                "eng": spacy.load("en_core_web_sm")}


def lazzy_download(uri, filename=None):
    if filename is None:
        path = Path(urlparse(uri).path)
        filename = Path(path.name)

    context = ssl._create_unverified_context()
    if not filename.is_file():
        with Halo(f"Downloading {filename}"):
            with urlopen(uri, context=context) as response:
                shutil.copyfileobj(response, open(filename, "wb"))

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
    
    with Halo(text=f'Extracting {original_filename}', spinner='dots'):
        with tarfile.open(archive, "r") as ifile:
            original_filename = ifile.getmember(original_filename)
            ifile.extractall(target_directory, [original_filename])

    return target_filename


def tokenize_with_spacy(input_files, output_fn, language):

    print("Tokenize data with spacy")
    nlp = SPACY_MODELS[language]

    total_size = sum(1 for _ in chain.from_iterable([open(fn) for fn in input_files]))

    with open(output_fn, "w") as ofile:
        for line in tqdm(chain.from_iterable(open(fn) for fn in input_files), total=total_size):
            tokenized_sentence = [token.text.strip() for token in nlp(line, parser=False, tagger=False, entity=False)]
            ofile.write(" ".join(tokenized_sentence))
            ofile.write("\n")


raw_data = create_directory("raw_data")
download_dir = create_directory("raw_data/downloads")
working_dir = create_directory("tmp")

newsco = lazzy_download("https://www.statmt.org/wmt15/training-parallel-nc-v10.tgz",
                        download_dir / "training-parallel-nc-v10.tgz")
common_crawl = lazzy_download("https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
                              download_dir / "training-parallel-commoncrawl.tgz")
europarl = lazzy_download("https://www.statmt.org/europarl/v7/fr-en.tgz",
                          download_dir / "fr-en.tgz")

fr_corpora = [
    extract_corpus(newsco, "news-commentary-v10.fr-en.fr", raw_data),
    extract_corpus(europarl, "europarl-v7.fr-en.fr", raw_data),
    extract_corpus(common_crawl, "commoncrawl.fr-en.fr", raw_data),
]

en_corpora = [
    extract_corpus(newsco, "news-commentary-v10.fr-en.en", raw_data),
    extract_corpus(europarl, "europarl-v7.fr-en.en", raw_data),
    extract_corpus(common_crawl, "commoncrawl.fr-en.en", raw_data),
]
              
#tokenize_with_spacy(fr_corpora, working_dir / "wmt15.tokenized.fra", "fra")
tokenize_with_spacy(en_corpora, working_dir / "wmt15.tokenized.eng", "eng")

import tarfile
import ssl
import shutil

from urllib.request import urlopen
from urllib.parse import urlparse
from pathlib import Path
from itertools import chain

import requests
import spacy

import sentencepiece as spm

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


def extract_corpus(archive, original_filename, target_filename):
    """
    Extract lazzily a file from an archive.

    For the moment, can only be applied to tar archives.
    """
    print(f"Extracting {original_filename}")

    target_filename = Path(target_filename)
    if target_filename.is_file():
        print(f"skip, {target_filename} already exists")
        return target_filename

    with Halo(text=f"Extracting {original_filename}", spinner="dots"):
        with tarfile.open(archive, "r") as ifile:
            original_file = ifile.getmember(original_filename)
            ifile.extractall(members=[original_file])

            original_filename = Path(original_filename)
            original_filename.rename(target_filename)
    
    return target_filename


def sgm2txt(input_filename, output_filename):

    with open(output_filename, "wt") as ofile:
        for line in open(input_filename):
            if "<seg" in line:
                ofile.write(line.split(">")[1].split("<")[0])
                ofile.write("\n")

    return output_filename


def merge_files(input_files, output_fn):
    total_size = sum(1 for _ in chain.from_iterable([open(fn, "rb") for fn in input_files]))

    with open(output_fn, "w") as ofile:
        for line in tqdm(
            chain.from_iterable(open(fn, "rb") for fn in input_files), total=total_size
        ):
            # Be carefull: we have to open the file in binary
            # otherwise python will take the \r that can appear within
            # a sentence as a newline breaking the parallel corpus
            # (this is for instance the case in the newscommentary
            # corpus)
            line = line.decode("utf-8")
            line = line.replace("\r", "")
            ofile.write(" ".join(line.split()))
            ofile.write("\n")

    return output_fn


def safe_open(filename):
    # Be carefull: we have to open the file in binary
    # otherwise python will take the \r that can appear within
    # a sentence as a newline breaking the parallel corpus
    # (this is for instance the case in the newscommentary
    # corpus)
    return (line.decode("utf-8").replace("\r", "") for line in open(filename, "rb"))


def tokenize_with_spacy(input_files, output_fn, language):

    print("Tokenize data with spacy")
    nlp = SPACY_MODELS[language]

    total_size = sum(1 for _ in chain.from_iterable([open(fn, "rb") for fn in input_files]))

    with open(output_fn, "w") as ofile:
        for line in tqdm(
            chain.from_iterable(open(fn, "rb") for fn in input_files), total=total_size
        ):
            # Be carefull: we have to open the file in binary
            # otherwise python will take the \r that can appear within
            # a sentence as a newline breaking the parallel corpus
            # (this is for instance the case in the newscommentary
            # corpus)
            line = line.decode("utf-8")
            line = line.replace("\r", "")
            tokenized_sentence = [
                token.text.strip() for token in nlp(line, disable=["parser", "ner"])
            ]
            ofile.write(" ".join(tokenized_sentence))
            ofile.write("\n")

    return output_fn

            
def bpe_tokenize(train_fn, vocab_size, model_prefix, tokenized_dir, files_to_tokenize, to_lower=False):

    model_prefix = Path(model_prefix)
    model_file = model_prefix.with_suffix(".model")
    model_directory = model_prefix.parent
    tokenized_dir = Path(tokenized_dir)

    tokenized_dir = create_directory(tokenized_dir)
    model_directory = create_directory(model_directory)

    if not model_file.is_file():
        spm.SentencePieceTrainer.train(input=train_fn,
                                       model_prefix=model_prefix,
                                       vocab_size=vocab_size)

    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    files_to_tokenize = [Path(f) for f in files_to_tokenize]
    
    for filename in files_to_tokenize:

        output_filename = tokenized_dir / f"{filename.stem}.bpe{filename.suffix}"
        size = sum(1 for _ in open(filename))
        
        if output_filename.is_file():
            print(f"do not tokenize {output_filename} has it already exists")
            continue
        
        with open(output_filename, "wt") as ofile:
            for line in tqdm(open(filename), total=size):
                if to_lower:
                    line = line.lower()
                ofile.write(" ".join(sp.encode(line, out_type=str)))
                ofile.write("\n")


def check_files(first, second):

    if type(first) != list:
        first = [first]

    if type(second) != list:
        second = [second]
        
    line_number_first = sum(1 for _ in chain.from_iterable([safe_open(f) for f in first]))
    line_number_second = sum(1 for _ in chain.from_iterable([safe_open(s) for s in second]))

    return line_number_second == line_number_first

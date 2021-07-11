"""
Prepare corpora for training a en-fr system on WMT'15 data.
"""

from download_corpus import create_directory, lazzy_download, extract_corpus, tokenize_with_spacy, sgm2txt, bpe_tokenize, merge_files, check_files

raw_data = create_directory("raw_data")
download_dir = create_directory("raw_data/downloads")
working_dir = create_directory("working_dir")

newsco = lazzy_download("https://www.statmt.org/wmt15/training-parallel-nc-v10.tgz",
                        download_dir / "training-parallel-nc-v10.tgz")
# common_crawl = lazzy_download("https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
#                               download_dir / "training-parallel-commoncrawl.tgz")
# europarl = lazzy_download("https://www.statmt.org/europarl/v7/fr-en.tgz",
#                           download_dir / "fr-en.tgz")

devsets = lazzy_download("https://www.statmt.org/wmt15/dev-v2.tgz",
                         download_dir / "dev.tgz")

fr_training_corpora = [
    extract_corpus(newsco, "news-commentary-v10.fr-en.fr", raw_data / "news-commentary-v10.fr-en.fr"),
#    extract_corpus(europarl, "europarl-v7.fr-en.fr", raw_data / "europarl-v7.fr-en.fr"),
#    extract_corpus(common_crawl, "commoncrawl.fr-en.fr", raw_data / "commoncrawl.fr-en.fr"),
]

en_training_corpora = [
    extract_corpus(newsco, "news-commentary-v10.fr-en.en", raw_data / "news-commentary-v10.fr-en.en"),
#    extract_corpus(europarl, "europarl-v7.fr-en.en", raw_data / "europarl-v7.fr-en.en"),
#    extract_corpus(common_crawl, "commoncrawl.fr-en.en", raw_data / "commoncrawl.fr-en.en"),
]

assert check_files(fr_training_corpora, en_training_corpora)

eng_test = extract_corpus(devsets, "dev/newstest2014-fren-ref.en.sgm", working_dir / "test.eng.sgm")
eng_test = sgm2txt(eng_test, working_dir / "test.eng")

fra_test = extract_corpus(devsets, "dev/newstest2014-fren-ref.fr.sgm", working_dir / "test.fra.sgm")
fra_test = sgm2txt(fra_test, working_dir / "test.fra")

eng_dev = extract_corpus(devsets, "dev/newstest2013.en", working_dir / "dev.eng")
fra_dev = extract_corpus(devsets, "dev/newstest2013.fr", working_dir / "dev.fra")

#fra_train_set = tokenize_with_spacy(fr_training_corpora, working_dir / "wmt15.tokenized.fra", "fra")
#eng_train_set = tokenize_with_spacy(en_training_corpora, working_dir / "wmt15.tokenized.eng", "eng")

fra_train_set = merge_files(fr_training_corpora, working_dir / "wmt15.tokenized.fra")
eng_train_set = merge_files(en_training_corpora, working_dir / "wmt15.tokenized.eng")

if not check_files(fra_train_set, eng_train_set):
    raise Exception
else:
    print(f"{fra_train_set} and {eng_train_set} have the same number of lines")


bpe_tokenize(train_fn=fra_train_set,
             files_to_tokenize=[fra_train_set, fra_dev, fra_test],
             tokenized_dir="wmt15/data",
             vocab_size=32_000,
             model_prefix="wmt15/model/fra_tokenizer",
             to_lower=True)

bpe_tokenize(train_fn=eng_train_set,
             files_to_tokenize=[eng_train_set, eng_dev, eng_test],
             tokenized_dir="wmt15/data",
             vocab_size=32_000,
             model_prefix="wmt15/model/eng_tokenizer",
             to_lower=True)

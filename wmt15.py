"""
Prepare corpora for training a en-fr system on WMT'15 data.
"""

from download_corpus import create_directory, lazzy_download, extract_corpus, tokenize_with_spacy

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
              
tokenize_with_spacy(fr_corpora, working_dir / "wmt15.tokenized.fra", "fra")
tokenize_with_spacy(en_corpora, working_dir / "wmt15.tokenized.eng", "eng")

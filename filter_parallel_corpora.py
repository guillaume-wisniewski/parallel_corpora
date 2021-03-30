import unicodedata
import argparse

from itertools import zip_longest
from collections import defaultdict, Counter
from pprint import pprint

import langid

from script_data import script_cat


def count_control_char(what):
    return sum(1 for ch in what if unicodedata.category(ch).startswith("C"))


parser = argparse.ArgumentParser()
parser.add_argument("--lower", action="store_true", default=False)
parser.add_argument("--src", required=True, type=argparse.FileType("rb"))
parser.add_argument("--tgt", required=True, type=argparse.FileType("rb"))
parser.add_argument("--max_sent_length", type=int, default=None)
parser.add_argument("--max_sentence", type=int, default=None)
parser.add_argument("--src_lang")
parser.add_argument("--tgt_lang")
parser.add_argument("--filter_lang", action="store_true")
parser.add_argument("--tgt_output", required=True, type=argparse.FileType("w"))
parser.add_argument("--src_output", required=True, type=argparse.FileType("w"))
args = parser.parse_args()

if args.filter_lang:
    assert args.src_lang is not None and args.tgt_lang is not None


stat = defaultdict(int)
old_sum = 0
for idx, (src, tgt) in enumerate(zip_longest(args.src, args.tgt)):

    src = src.decode("utf-8")
    tgt = tgt.decode("utf-8")
    
    if args.max_sentence is not None and idx >= args.max_sentence:
        break

    if src is None or tgt is None:
        print(f"src= {src}")
        print(f"tgt= {tgt}")
        break

    src = src.strip() if not args.lower else src.lower().strip()
    tgt = tgt.strip() if not args.lower else tgt.lower().strip()

    if not src or not tgt:
        stat["empty_sentence"] += 1
        continue

    if count_control_char(src) != 0 or count_control_char(tgt) != 0:
        stat["contains_control_char"] += 1
        continue

    if args.max_sent_length is not None and (len(src.split()) > args.max_sent_length or len(tgt.split()) > args.max_sent_length):
        stat["too_long"] += 1
        continue
    
    src_counts = Counter(script_cat(ch)[0] for ch in src)
    tgt_counts = Counter(script_cat(ch)[0] for ch in tgt)
    
    if sum(src_counts[w] for w in ("Inherited", "Common", "Latin")) - src.count("<") - src.count(">") < .9 * len(src) or \
       sum(tgt_counts[w] for w in ("Inherited", "Common", "Latin")) - src.count("<") - src.count(">") < .9 * len(tgt) or \
       tgt_counts["Latin"] < len(tgt) / 2 or\
       src_counts["Latin"] < len(src) / 2:
        stat["not_enough_latin"] += 1
        continue

    if args.filter_lang and ( langid.classify(src)[0] != args.src_lang or langid.classify(tgt)[0] != args.tgt_lang):
        stat["wrong_lang"] += 1
        continue

    stat["ok"] += 1
    args.tgt_output.write(f"{tgt}\n")
    args.src_output.write(f"{src}\n")

    if idx % 10_000 == 0:
        print(f"{idx:,}")
    
pprint(stat)


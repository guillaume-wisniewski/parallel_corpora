Script to filter parallel corpora

Remove parallel sentences:
- that are too long;
- that contain too many control characters or letters that are not in the Latin script;
- which are not in the correct language (according to `langid` ).

The rules implemented rules are mainly based on [Kenneth Heafield's preprocess tool](https://github.com/kpu/preprocess).

To install dependencies:
```
python -m venv local
./local/bin/pip install -r requirements.txt
```

See `wmt15.py` for an example.

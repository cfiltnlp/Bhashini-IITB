from flask import Flask, jsonify, request
import requests
import json

from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2



## Normalize
factory=IndicNormalizerFactory()
normalizer_hi=factory.get_normalizer("hi")
## BPE
codes_hi = codecs.open("hi-mr-iitb-pti/labse-pti/bpe-codes/codes.hi", encoding='utf-8')
bpe_hi = BPE(codes_hi)

## Translate
translator_himr = ctranslate2.Translator("hi-mr-iitb-pti/model_ct2/", inter_threads=4, intra_threads=1)

# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, "hi")

# Normalize
inp_lines = [normalizer_hi.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_hi.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_himr.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]


print(out_lines)
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

normalizer_mr=factory.get_normalizer("mr")
## BPE
codes_mr = codecs.open("mr-hi-iitb-pti/bpe-codes/codes.mr", encoding='utf-8')
bpe_mr = BPE(codes_mr)
## Translate
translator_mrhi = ctranslate2.Translator("mr-hi-iitb-pti/labse-pti/model_ct2/", inter_threads=4, intra_threads=1)
paras = ""

# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, "mr")

# Normalize
inp_lines = [normalizer_mr.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_mr.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_mrhi.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

print(out_lines)

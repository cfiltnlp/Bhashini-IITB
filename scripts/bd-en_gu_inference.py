from mosestokenizer import MosesDetokenizer
from indicnlp.tokenize import indic_tokenize, sentence_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2

# Normalize
factory=IndicNormalizerFactory()
normalizer_bd=factory.get_normalizer("hi")

# BPE
codes_bden_bd = codecs.open("bpe-codes/bd_code.txt", encoding='utf-8')
bpe_codes_bden_bd = BPE(codes_bden_bd)

# Translator
translator_bden = ctranslate2.Translator("model_deploy/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)

# Detokenizer
detokenize_en = MosesDetokenizer('en')


paras = 'बेयो मोनसे आनजाद बिजिरथि। बेयो मोनसे गुबुन आनजाद बिजिरथि।'

# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, 'hi')

# Normalize
inp_lines = [normalizer_bd.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_codes_bden_bd.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_bden.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [detokenize_en(line.split(" ")) for line in out_lines]

# Capitalize
out_lines = [line.capitalize() for line in out_lines]

print(out_lines)
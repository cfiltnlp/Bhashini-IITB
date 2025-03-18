from mosestokenizer import MosesDetokenizer
from indicnlp.tokenize import indic_tokenize, sentence_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2


# Normalize
factory=IndicNormalizerFactory()
normalizer_ne=factory.get_normalizer("ne")

# BPE
codes_ne_en__ne = codecs.open("bpe-codes/codes.ne", encoding='utf-8')
bpe_ne_en__ne = BPE(codes_ne_en__ne)

# Translator
translator_neen = ctranslate2.Translator("model_deploy",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)

# Detokenizer
detokenize_en = MosesDetokenizer('en')


paras = 'यो एक परीक्षामूलक शास्त्र हो। यो एक परीक्षा हो।'

# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, 'ne')

# Normalize
inp_lines = [normalizer_ne.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_ne_en__ne.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_neen.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [detokenize_en(line.split(" ")) for line in out_lines]

# Capitalize
out_lines = [line.capitalize() for line in out_lines]

print(out_lines)
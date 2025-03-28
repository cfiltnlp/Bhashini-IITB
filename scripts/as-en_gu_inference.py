from mosestokenizer import MosesDetokenizer
from indicnlp.tokenize import indic_tokenize, sentence_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2

# Normalize
factory=IndicNormalizerFactory()
normalizer_as=factory.get_normalizer("as")

## BPE
codes_asen_as = codecs.open("bpe-codes/codes.as", encoding='utf-8')
bpe_codes_asen_as = BPE(codes_asen_as)

# Translator
translator_asen = ctranslate2.Translator("model_deploy/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)

# Detokenizer
detokenize_en = MosesDetokenizer('en')


paras = 'এয়া এক পৰীক্ষামূলক শাস্তি। এয়া হৈছে আন এক পৰীক্ষা।'


# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, 'as')

# Normalize
inp_lines = [normalizer_as.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_codes_asen_as.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_asen.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [detokenize_en(line.split(" ")) for line in out_lines]

# Capitalize
out_lines = [line.capitalize() for line in out_lines]

print(out_lines)


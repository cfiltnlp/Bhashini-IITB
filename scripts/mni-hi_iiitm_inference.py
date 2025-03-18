from indicnlp.tokenize import indic_tokenize, sentence_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2

# Normalizer
factory=IndicNormalizerFactory()
normalizer_mni=factory.get_normalizer("mP")

# BPE
codes_hi_mni_mni_codes = codecs.open("codes/codes.mni", encoding='utf-8')
bpe_hi_mni_mni_bpe = BPE(codes_hi_mni_mni_codes)

# Translator
translator_mni_hi = ctranslate2.Translator("translator_ct/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)


paras = 'ꯃꯁꯤ ꯑꯅꯤꯁꯨꯕ ꯇꯦꯁ꯭ꯇ ꯆꯩꯔꯥꯛ ꯑꯃꯅꯤ। ꯃꯁꯤ ꯇꯦꯁ꯭ꯇ ꯋꯥꯍꯩꯅꯤ ꯫'


# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, 'mP')

# Normalize
inp_lines = [normalizer_mni.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_hi_mni_mni_bpe.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = bpe_hi_mni_mni_bpe.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [indic_detokenize.trivial_detokenize(line) for line in out_lines]

print(out_lines)

from mosestokenizer import MosesDetokenizer
from indicnlp.tokenize import indic_tokenize, sentence_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2

# Normalize
factory=IndicNormalizerFactory()
normalizer_mni= factory.get_normalizer("mP")

# BPE
codes_en_mni_mni_codes = codecs.open("codes/codes.mni", encoding='utf-8')
bpe_en_mni_mni_bpe = BPE(codes_en_mni_mni_codes)

# Translator
translator_mni_en = ctranslate2.Translator("model_deploy/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)


paras = 'ꯃꯁꯤ ꯑꯅꯤꯁꯨꯕ ꯇꯦꯁ꯭ꯇ ꯆꯩꯔꯥꯛ ꯑꯃꯅꯤ। ꯃꯁꯤ ꯇꯦꯁ꯭ꯇ ꯋꯥꯍꯩꯅꯤ ꯫'


# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, 'mP')

# Detokenizer
detokenize_en = MosesDetokenizer('en')


# Normalize
inp_lines = [normalizer_mni.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_en_mni_mni_bpe.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_mni_en.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [detokenize_en(line.split(" ")) for line in out_lines]

print(out_lines)

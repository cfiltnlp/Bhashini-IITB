from indicnlp.tokenize import sentence_tokenize, indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import codecs
from subword_nmt.apply_bpe import BPE
import ctranslate2


# Normalize
factory=IndicNormalizerFactory()
normalizer_kk=factory.get_normalizer("kK")

# BPE
codes_konkani_hi_code = codecs.open("bpe-codes/codes.go", encoding='utf-8')
bpe_konkani_hi_bpe = BPE(codes_konkani_hi_code)

## Translator
translator_konkani_hi = ctranslate2.Translator("model_deploy/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)


paras = 'म्हाका सफरचंद खावपाक आवडटा. हें झाड खूब ऊंच आसता.'


# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, "kK")

# Normalize
inp_lines = [normalizer_kk.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_konkani_hi_bpe.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_konkani_hi.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [indic_detokenize.trivial_detokenize(line) for line in out_lines]

print(out_lines)
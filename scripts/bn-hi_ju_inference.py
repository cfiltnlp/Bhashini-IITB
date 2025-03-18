from indicnlp.tokenize import sentence_tokenize, indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import codecs
from subword_nmt.apply_bpe import BPE
import ctranslate2


## Normalize
factory=IndicNormalizerFactory()
normalizer_hi=factory.get_normalizer("bn")

# BPE
codes_bn_hi_ju_bn = codecs.open("bpe-codes/codes.bn", encoding='utf-8')
bpe_bn_hi_ju_bn = BPE(codes_bn_hi_ju_bn)

# Translator
translator_bnhi_ju = ctranslate2.Translator("model_deploy/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)


paras = 'আমি আপেল খেতে পছন্দ করি। গাছটি অনেক লম্বা।'


# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, "bn")

# Normalize
inp_lines = [normalizer_hi.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_bn_hi_ju_bn.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_bnhi_ju.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [indic_detokenize.trivial_detokenize(line) for line in out_lines]

print(out_lines)
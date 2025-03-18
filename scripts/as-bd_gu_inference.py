from indicnlp.tokenize import sentence_tokenize, indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import codecs
from subword_nmt.apply_bpe import BPE
import ctranslate2


# Normalize
factory=IndicNormalizerFactory()
normalizer_as=factory.get_normalizer("as")

# BPE
codes = codecs.open("bpe-codes/codes.as", encoding='utf-8')
bpe = BPE(codes)

# Translator
translator = ctranslate2.Translator("model_deploy/",
    # compute_type="int8"
    )


paras = 'এই আই এছ এ পৰীক্ষাৰ বাক্য। এই আই এছ এ দ্বিতীয় পৰীক্ষা বাক্য।'


# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, "as")

# Normalize
inp_lines = [normalizer_as.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [indic_detokenize.trivial_detokenize(line) for line in out_lines]

print(out_lines)
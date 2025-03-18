import ctranslate2
from mosestokenizer import MosesSentenceSplitter, MosesTokenizer
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import codecs
from subword_nmt.apply_bpe import BPE

## Normalize
factory=IndicNormalizerFactory()
normalizer=factory.get_normalizer("hi")

## BPE
codes = codecs.open("bpe-codes/codes.hi", encoding='utf-8')
bpe = BPE(codes)

## Translate
translator = ctranslate2.Translator("model_deploy/",
    # compute_type="int8"
    )

inp_lines = ['मुझे सेब खाना पसंद है ।', 'पेड़ बहुत ऊंचा है ।']

# Normalize
inp_lines = [normalizer.normalize(line) for line in inp_lines]

# Tokenize
inp_lines = [' '.join(indic_tokenize.trivial_tokenize(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]
print(out_lines)

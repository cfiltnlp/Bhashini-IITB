from mosestokenizer import MosesSentenceSplitter, MosesTokenizer
from indicnlp.tokenize import indic_detokenize

import codecs
from subword_nmt.apply_bpe import BPE

import ctranslate2


# Sentence Splitter
splitsents_en = MosesSentenceSplitter('en')

# Tokenizer
tokenize_en = MosesTokenizer('en')

# BPE
codes_en_mni_en_codes = codecs.open("codes/codes.en", encoding='utf-8')
bpe_en_mni_en_bpe = BPE(codes_en_mni_en_codes)

# Translator
translator_en_mni = ctranslate2.Translator("translator_ct/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)


paras = "This is a test sentence. This is another sentence."


# Split Sentences
inp_lines = splitsents_en([paras.strip('\n')])

# Lowercase
inp_lines = [line.lower() for line in inp_lines]

# Tokenize
inp_lines = [' '.join(tokenize_en(line)) for line in inp_lines]

# Apply BPE
inp_lines = [bpe_en_mni_en_bpe.process_line(line).split(" ") for line in inp_lines]

# Translate
out_lines = translator_en_mni.translate_batch(inp_lines, beam_size=5, max_batch_size=16)

# Remove BPE
out_lines = [(' '.join(line.hypotheses[0]) + " ").replace("@@ ", "") for line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

# Detokenize
out_lines = [indic_detokenize.trivial_detokenize(line) for line in out_lines]

print(out_lines)

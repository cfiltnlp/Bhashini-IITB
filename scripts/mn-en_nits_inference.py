import sentencepiece as sp
import ctranslate2
from indicnlp.tokenize import sentence_tokenize


# Sentencepiece processor
sp_mnen_nits = sp.SentencePieceProcessor(model_file='train.model')

# Translator
translator_mnen_nits = ctranslate2.Translator("model_deploy/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)


paras = 'ꯃꯁꯤ ꯑꯅꯤꯁꯨꯕ ꯇꯦꯁ꯭ꯇ ꯆꯩꯔꯥꯛ ꯑꯃꯅꯤ। ꯃꯁꯤ ꯇꯦꯁ꯭ꯇ ꯋꯥꯍꯩꯅꯤ ꯫'


# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, 'mP')

# Apply sentencepiece
inp_lines = sp_mnen_nits.encode_as_pieces(inp_lines)

# Translate
out_lines = translator_mnen_nits.translate_batch(inp_lines, beam_size=5, max_batch_size=16)
out_lines = [out_lines[i].hypotheses[0] for i in range(len(out_lines))]

# Remove sentencepiece
out_lines = [sp_mnen_nits.decode(out_line).replace(chr(9601), " ") for out_line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

print(out_lines)

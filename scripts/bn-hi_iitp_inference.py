from indicnlp.tokenize import sentence_tokenize
import sentencepiece as sp
import ctranslate2


# Sentencepiece processor
sp_bnhi_iitp = sp.SentencePieceProcessor(model_file='train.model')

# Translator
translator_bnhi_iitp = ctranslate2.Translator("model_deploy/",
    # compute_type="int8",
    inter_threads=4, intra_threads=1)


paras = 'আমি আপেল খেতে পছন্দ করি। গাছটি অনেক লম্বা।'


# Split Sentences
inp_lines = sentence_tokenize.sentence_split(paras, "bn")

# Apply sentencepiece
inp_lines = sp_bnhi_iitp.encode_as_pieces(inp_lines)

# Translate
out_lines = translator_bnhi_iitp.translate_batch(inp_lines, beam_size=5, max_batch_size=16)
out_lines = [out_lines[i].hypotheses[0] for i in range(len(out_lines))]

# Remove sentencepiece
out_lines = [sp_bnhi_iitp.decode(out_line) for out_line in out_lines]

# Post Processing
out_lines = [line.replace('"', '').replace("u200d", "").strip() for line in out_lines]

print(out_lines)
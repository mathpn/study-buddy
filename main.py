# %%

import numpy as np
import ollama
import pymupdf4llm
from docling.document_converter import DocumentConverter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from markitdown import MarkItDown

# %%

doc = pymupdf4llm.to_markdown("input_test.pdf")
with open("extracted_pymupdf.md", "w") as f:
    f.write(doc)

# %%

md = MarkItDown()
result = md.convert("input_test.pdf")
with open("extracted_markitdown.md", "w") as f:
    f.write(result.markdown)
# %%

converter = DocumentConverter()
result = converter.convert("input_test.pdf")
with open("extracted_docling.md", "w") as f:
    f.write(result.document.export_to_markdown())


# %%

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("input_test.pdf")
text, _, images = text_from_rendered(rendered)
with open("extracted_marker.md", "w") as f:
    f.write(text)

# %%

doc = """
In computer science and telecommunications, Hamming codes are a family of linear error-correcting codes. Hamming codes can detect one-bit and two-bit errors, or correct one-bit errors without detection of uncorrected errors. By contrast, the simple parity code cannot correct errors, and can detect only an odd number of bits in error. Hamming codes are perfect codes, that is, they achieve the highest possible rate for codes with their block length and minimum distance of three.[1] Richard W. Hamming invented Hamming codes in 1950 as a way of automatically correcting errors introduced by punched card readers. In his original paper, Hamming elaborated his general idea, but specifically focused on the Hamming(7,4) code which adds three parity bits to four bits of data.[2]

In mathematical terms, Hamming codes are a class of binary linear code. For each integer r ≥ 2 there is a code-word with block length n = 2r − 1 and message length k = 2r − r − 1. Hence the rate of Hamming codes is R = k / n = 1 − r / (2r − 1), which is the highest possible for codes with minimum distance of three (i.e., the minimal number of bit changes needed to go from any code word to any other code word is three) and block length 2r − 1. The parity-check matrix of a Hamming code is constructed by listing all columns of length r that are non-zero, which means that the dual code of the Hamming code is the shortened Hadamard code, also known as a Simplex code. The parity-check matrix has the property that any two columns are pairwise linearly independent.

Due to the limited redundancy that Hamming codes add to the data, they can only detect and correct errors when the error rate is low. This is the case in computer memory (usually RAM), where bit errors are extremely rare and Hamming codes are widely used, and a RAM with this correction system is an ECC RAM (ECC memory). In this context, an extended Hamming code having one extra parity bit is often used. Extended Hamming codes achieve a Hamming distance of four, which allows the decoder to distinguish between when at most one one-bit error occurs and when any two-bit errors occur. In this sense, extended Hamming codes are single-error correcting and double-error detecting, abbreviated as SECDED.

Richard Hamming, the inventor of Hamming codes, worked at Bell Labs in the late 1940s on the Bell Model V computer, an electromechanical relay-based machine with cycle times in seconds. Input was fed in on punched paper tape, seven-eighths of an inch wide, which had up to six holes per row. During weekdays, when errors in the relays were detected, the machine would stop and flash lights so that the operators could correct the problem. During after-hours periods and on weekends, when there were no operators, the machine simply moved on to the next job.

Hamming worked on weekends, and grew increasingly frustrated with having to restart his programs from scratch due to detected errors. In a taped interview, Hamming said, "And so I said, 'Damn it, if the machine can detect an error, why can't it locate the position of the error and correct it?'".[3] Over the next few years, he worked on the problem of error-correction, developing an increasingly powerful array of algorithms. In 1950, he published what is now known as Hamming code, which remains in use today in applications such as ECC memory.
"""

# %%

embedding_model = "nomic-embed-text"

chunks = []
vectors = []


def extract_embedding(chunk):
    embedding = ollama.embed(model=embedding_model, input=chunk)["embeddings"]
    return embedding


for i, chunk in enumerate(l.strip() for l in doc.split("\n") if l.strip()):
    emb = extract_embedding(chunk)
    chunks.append(chunk)
    vectors.append(np.array(emb))
    print(f"Added chunk {i+1} to vectors")

vectors = np.concatenate(vectors)
vectors.shape

# %%


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def retrieve(query: str, top_n: int = 3):
    query_embedding = np.array(
        ollama.embed(model=embedding_model, input=query)["embeddings"][0]
    )

    similarities = cosine_similarity(query_embedding, vectors)
    top_idx = np.argsort(-similarities)[:top_n]
    return [(chunks[idx], similarities[idx]) for idx in top_idx]


retrieve("ECC memory.")

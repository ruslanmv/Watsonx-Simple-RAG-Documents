import pdfplumber
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Avoid None pages
                full_text += page_text + "\n"
    return full_text

def chunk_text_by_tokens(text, model_name="sentence-transformers/all-MiniLM-L6-v2", max_tokens=510, overlap_tokens=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(text_input):
        return len(tokenizer.encode(text_input))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=count_tokens,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    # Debug info for verification:
    print(f"✅ Total Chunks Created: {len(chunks)}\n")

    # Print chunk sizes in tokens to verify correct chunking
    for i, chunk in enumerate(chunks):
        tokens_len = count_tokens(chunk)
        print(f"Chunk {i+1} token length: {tokens_len}")

    # Verify overlap: print last 30 chars of current chunk and first 30 chars of next chunk
    print("\n🔍 Overlap check between consecutive chunks:")
    for i in range(len(chunks) - 1):
        overlap_current = chunks[i][-30:]
        overlap_next = chunks[i+1][:30]
        print(f"Chunk {i+1} end: '{overlap_current}'")
        print(f"Chunk {i+2} start: '{overlap_next}'")
        print("---")

    # Print a preview of first chunk (first 1000 characters)
    print("\n📄 Preview of First Chunk (first 1000 chars):\n")
    print(chunks[0][:1000])

    return chunks

# ======== RUN BELOW THIS ===========

pdf_file = "sample.pdf"  # 👈 Rename this to your actual file name
text = extract_text_from_pdf(pdf_file)

chunks = chunk_text_by_tokens(text, max_tokens=510, overlap_tokens=50)

print("\n✅ Chunking completed successfully and output looks stable!")

"""
ingest.py - chromadb default embedding (bepul, kichik)
"""
import os
from docx import Document
import chromadb

DOCS_FOLDER = "./docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_docs")


def read_docx(filepath):
    doc = Document(filepath)
    return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())


def split_into_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_all_docs():
    docx_files = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".docx")]
    if not docx_files:
        print("❌ docs/ papkasida .docx fayl topilmadi!")
        return

    print(f"📂 {len(docx_files)} ta fayl topildi...\n")
    total = 0

    for filename in docx_files:
        print(f"📄 O'qilmoqda: {filename}")
        text = read_docx(os.path.join(DOCS_FOLDER, filename))
        chunks = split_into_chunks(text)
        print(f"   → {len(chunks)} ta chunk")

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            cid = f"{filename}_chunk_{i}"
            if not collection.get(ids=[cid])["ids"]:
                ids.append(cid)
                docs.append(chunk)
                metas.append({"source": filename, "chunk_index": i})

        if ids:
            collection.add(ids=ids, documents=docs, metadatas=metas)
            total += len(ids)
        print(f"   ✅ Saqlandi\n")

    print(f"🎉 Tayyor! {total} ta chunk saqlandi.")


if __name__ == "__main__":
    ingest_all_docs()
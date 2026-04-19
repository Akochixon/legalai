"""
rag.py - chromadb default embedding + Groq LLM
"""
import os
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="legal_docs")


def search_relevant_chunks(question, top_k=5):
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta.get("source", "Noma'lum")})
    return chunks


def generate_answer(question, chunks):
    context = "\n\n---\n\n".join(f"[Manba: {c['source']}]\n{c['text']}" for c in chunks)

    system_prompt = """Siz O'zbekiston qonunchiligi bo'yicha mutaxassis huquqiy yordamchisiz.
Foydalanuvchi savollariga faqat berilgan qonun matnlari asosida javob bering.
Javobingiz O'zbek tilida, oddiy va tushunarli bo'lsin.
Qonun moddalari va manbalarga havola qiling.
Agar ma'lumot topilmasa, halol aytib bering."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Qonun matnlari:\n{context}\n\nSavol: {question}"}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": list(set(c["source"] for c in chunks))
    }


def ask(question):
    return generate_answer(question, search_relevant_chunks(question))


if __name__ == "__main__":
    q = "Mas'uliyati cheklangan jamiyat nima?"
    result = ask(q)
    print(f"✅ Javob:\n{result['answer']}")
    print(f"📚 Manbalar: {', '.join(result['sources'])}")
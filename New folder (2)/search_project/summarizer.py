from db import SessionLocal, Paper
from transformers import pipeline

def chunk_text(text, max_chunk_size=500):
    """Split text into chunks of roughly max_chunk_size words for summarization."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i + max_chunk_size])
        chunks.append(chunk)
    return chunks

def summarize_papers():
    db = SessionLocal()
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # set device=0 for GPU, or -1 for CPU

    papers = db.query(Paper).all()

    for paper in papers:
        if paper.abstract:
            
            chunks = chunk_text(paper.abstract, max_chunk_size=500)
            summaries = []

            for chunk in chunks:
                input_len = len(chunk.split())
                
                max_len = min(150, max(40, input_len // 3))
                min_len = max(20, max_len // 2)

                summary = summarizer(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False
                )[0]['summary_text']

                summaries.append(summary)

            full_summary = " ".join(summaries)
            print(f"Summary for '{paper.title}': {full_summary}\n")

    db.close()

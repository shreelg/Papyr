# from sentence_transformers import SentenceTransformer
# import faiss
# from transformers import pipeline
# from db import SessionLocal, Paper
# import numpy as np
# import re

# def simple_sentence_tokenize(text):
#     # Split text into sentences on ., !, ? followed by space/newline
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     return [s.strip() for s in sentences if s.strip()]

# class PaperQA:
#     def __init__(self):
#         self.db = SessionLocal()
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.generator = pipeline("text2text-generation", model="google/flan-t5-large")

#         self.passages = []
#         self.passage_map = []  # maps passage idx to (paper_id, passage_text)
#         self.embeddings = None
#         self.index = None

#     def load_papers_from_db(self):
#         # Load papers with abstracts
#         self.papers = self.db.query(Paper).filter(Paper.abstract.isnot(None)).all()
#         if not self.papers:
#             raise ValueError("No paper abstracts found in the database.")

#         self.passages.clear()
#         self.passage_map.clear()

#         # Split abstracts into sentences and build mappings
#         for paper in self.papers:
#             sentences = simple_sentence_tokenize(paper.abstract)
#             for sent in sentences:
#                 self.passages.append(sent)
#                 self.passage_map.append((paper.id, sent))

#         # Generate embeddings for all passages
#         self.embeddings = self.embedding_model.encode(self.passages, convert_to_numpy=True)
#         if self.embeddings is None or len(self.embeddings) == 0:
#             raise ValueError("Failed to generate embeddings.")

#         # Build FAISS index for fast similarity search
#         dim = self.embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(dim)
#         self.index.add(self.embeddings)

#     def answer_question(self, question: str, paper_id: int):
#         if self.index is None:
#             raise RuntimeError("Index not initialized. Call load_papers_from_db() first.")

#         # Filter indices corresponding to the requested paper
#         relevant_indices = [i for i, (pid, _) in enumerate(self.passage_map) if pid == paper_id]
#         if not relevant_indices:
#             return "Paper not found or no passages available."

#         paper_embeddings = self.embeddings[relevant_indices]

#         # Temporary FAISS index for only the paper's passages
#         dim = paper_embeddings.shape[1]
#         paper_index = faiss.IndexFlatL2(dim)
#         paper_index.add(paper_embeddings)

#         # Embed the question
#         question_emb = self.embedding_model.encode([question], convert_to_numpy=True)

#         # Retrieve top 3 relevant passages (or fewer if not enough)
#         k = min(3, len(relevant_indices))
#         _, I = paper_index.search(question_emb, k)

#         retrieved_passages = [self.passage_map[relevant_indices[i]][1] for i in I[0]]
#         if not retrieved_passages:
#             return "No relevant passages found for this question."

#         context_text = "\n".join(retrieved_passages)

#         input_text = (
#     "You are a knowledgeable research assistant specializing in scientific papers. "
#     "Based on the following extracted sentences from a research paper, provide a clear and concise answer to the question. "
#     "If the information is not available in the context, respond with: 'No relevant information found in the paper.'\n\n"
#     f"Context:\n{context_text}\n\n"
#     f"Question: {question}\n"
#     "Answer:"
# )

#         try:
#             output = self.generator(input_text, max_length=150, do_sample=False)[0]['generated_text']
#         except Exception as e:
#             return f"Error generating answer: {str(e)}"

#         return output


from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from db import SessionLocal, Paper
import re

def simple_sentence_tokenize(text):
    """Split text into sentences based on punctuation followed by whitespace."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

class PaperQA:
    def __init__(self):
        self.db = SessionLocal()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.generator = pipeline("text2text-generation", model="google/flan-t5-large")

        self.passages = []
        self.passage_map = []  # List of tuples: (paper_id, passage_text)
        self.embeddings = None
        self.index = None

    def load_papers_from_db(self):
        """Load abstracts from the database, tokenize, embed, and build FAISS index."""
        self.papers = self.db.query(Paper).filter(Paper.abstract.isnot(None)).all()
        if not self.papers:
            raise ValueError("No paper abstracts found in the database.")

        self.passages.clear()
        self.passage_map.clear()

        # Tokenize abstracts into sentences and map them to paper IDs
        for paper in self.papers:
            sentences = simple_sentence_tokenize(paper.abstract)
            for sentence in sentences:
                self.passages.append(sentence)
                self.passage_map.append((paper.id, sentence))

        # Generate embeddings for all passages
        self.embeddings = self.embedding_model.encode(self.passages, convert_to_numpy=True)
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("Failed to generate embeddings.")

        # Create and populate FAISS index for similarity search
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def answer_question(self, question: str, paper_id: int):
        """Answer a question about a specific paper using nearest passages and text generation."""
        if self.index is None:
            raise RuntimeError("Index not initialized. Call load_papers_from_db() first.")

        # Get indices of passages related to the requested paper
        relevant_indices = [i for i, (pid, _) in enumerate(self.passage_map) if pid == paper_id]
        if not relevant_indices:
            return "Paper not found or no passages available."

        # Extract embeddings of relevant passages
        paper_embeddings = self.embeddings[relevant_indices]

        # Build a temporary FAISS index for the paper's passages
        dim = paper_embeddings.shape[1]
        paper_index = faiss.IndexFlatL2(dim)
        paper_index.add(paper_embeddings)

        # Embed the question
        question_emb = self.embedding_model.encode([question], convert_to_numpy=True)

        # Retrieve top 3 relevant passages
        k = min(3, len(relevant_indices))
        _, retrieved_indices = paper_index.search(question_emb, k)

        retrieved_passages = [self.passage_map[relevant_indices[i]][1] for i in retrieved_indices[0]]
        if not retrieved_passages:
            return "No relevant passages found for this question."

        context_text = "\n".join(retrieved_passages)

        input_text = (
            "You are a knowledgeable research assistant specializing in scientific papers. "
            "Search the paper up and get the answer. "
            "If the information is not available in the context, respond with: 'No relevant information found in the paper.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        try:
            output = self.generator(input_text, max_length=150, do_sample=False)[0]['generated_text']
        except Exception as e:
            return f"Error generating answer: {str(e)}"

        return output

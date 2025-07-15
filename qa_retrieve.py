from sentence_transformers import SentenceTransformer
import torch


model = SentenceTransformer('all-MiniLM-L6-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def embed_chunks(chunks):
    
    return model.encode(chunks, convert_to_tensor=True, device=device)

    

def top_k(question, doc_embeddings, doc_chunks, k=3):
    question_embedding = model.encode([question], convert_to_tensor=True, device=device)
    cosine_scores = torch.nn.functional.cosine_similarity(question_embedding, doc_embeddings)


    k = min(k, len(doc_chunks))

    top_indices = torch.topk(cosine_scores, k=k).indices



    return [doc_chunks[idx] for idx in top_indices]



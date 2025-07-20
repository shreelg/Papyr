# Papyr

**[Video Demo](https://drive.google.com/file/d/1BRWY4qzByI3Mx-w6rEsQRV4z7Emam9Ks/view?usp=sharing)**

Papyr is a research-focused text summarization and question-answering tool, designed for fast, domain-specific information retrieval. Built for researchers and professionals, it summarizes academic texts and provides precise answers using state-of-the-art NLP models.


### Summarization Engine

- Powered by BART-large-CNN, fine-tuned on a custom dataset of 4,000 research papers.
- Achieved a ROUGE score of 0.9127.
- Converts academic and technical documents into concise, human-readable summaries.

### QA Bot

- Uses **Google FLAN-T5** to generate natural language answers in sentence or paragraph formats. (qa_generate.py)
- Supports fast retrieval using a **MiniLM-L6** sentence-transformer model for embedding and vectorizing document chunks. (qa_retrieve.py)
- Embedded chunks are stored for efficient similarity-based retrieval.
- Includes a **custom Top-K retrieval system** with optimized chunking and storage logic to make sure relevant answers are surfaced quickly.
 embedded chunks for fast similarity-based retrieval.


## File structure
<img width="231" height="323" alt="image" src="https://github.com/user-attachments/assets/c51e3bcf-35b9-49ea-8308-c9d449c271dc" />


## Deploy

1. Clone the repository:
```bash
git clone [https://github.com/your-username/papyr.git](https://github.com/shreelg/Papyr.git)
cd papyr

2. docker build -t papyr-app .


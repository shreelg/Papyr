# Papyr

**leverages web scraping and APIs to retrieve research papers from multiple sources and uses a fine-tuned Hugging Face model to summarize them efficiently. The tool is designed for researchers and students to easily access and digest academic papers.


### In return_information.py: 
- Uses web scraping and APIs to collect data from CORE, arxiv, PubMed, Semantic Scholar:
  - Title
  - Authors
  - Publication date
  - Paper links
- Supports topic-based queries to gather relevant papers.
- Retrieves full-text content from paper links.
- Handles plain text and HTML/PDF parsing where applicable.

### In model.py
- Hosted and fine-tuned a Hugging Face [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) model.
- Personally trained the model for **6 epochs** on a custom dataset of **4,000 research papers** (full texts and abstracts).
- Capable of summarizing large research papers (up to **~45,000 characters** / **~11k tokens**).
- This setup achieved a **ROUGE score of 0.9127** on the validation set.


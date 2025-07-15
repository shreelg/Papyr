from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import uvicorn

from summarizer_qa.qa_retrieve import top_k, embed_chunks
from summarizer_qa.qa_generate import generate_answer


from fastapi import UploadFile, File

app = FastAPI()

model_path = "bart-model_data-summary"  # Your local model folder
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model.config.forced_bos_token_id = 0



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

templates = Jinja2Templates(directory="templates")  # your templates folder

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "input_text": "", "summary": ""})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, text: str = Form(...)):
    try:
        inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]  # number of tokens

        max_summary_len = min(max(30, int(input_len * 0.6)), 250)
        min_summary_len = max(30, int(input_len * 0.1))

        summary_ids = model.generate(
    inputs["input_ids"],
    max_length=max_summary_len,
    min_length=min_summary_len,
    length_penalty=1.0,
    temperature=2.1,         
    do_sample=True,
    top_k=30,              
    top_p=0.9,               
    num_beams=4,             # beam search 
    no_repeat_ngram_size=3,
    early_stopping=True
)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        summary = f"Error during summarization: {e}"

    return templates.TemplateResponse("index.html", {"request": request, "input_text": text, "summary": summary})





# @app.post("/upload", response_class=HTMLResponse)
# async def upload_file(request: Request, file: UploadFile = File(...)):
#     try:
#         # Read file content as text
#         content = await file.read()
#         text = content.decode("utf-8")

#         # Tokenize and summarize (same as your existing summarize endpoint)
#         inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
#         inputs = {k: v.to(device) for k, v in inputs.items()}

#         summary_ids = model.generate(
#             inputs["input_ids"],
#             num_beams=4,
#             max_length=250,
#             min_length=50,
#             early_stopping=True,
#             no_repeat_ngram_size=3,
#             length_penalty=2.0,
#         )
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     except Exception as e:
#         summary = f"Error processing file: {e}"
#         text = ""

#     return templates.TemplateResponse("index.html", {"request": request, "input_text": text, "summary": summary})

@app.post("/qabot", response_class=HTMLResponse)
async def qa_bot(request: Request, question: str = Form(...), text: str = Form(...)):
    doc_chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not doc_chunks:
        answer = "No valid text chunks found in document."
        return templates.TemplateResponse("index.html", {
            "request": request,
            "input_text": text,
            "question": question,
            "answer": answer
        })

    doc_embeddings = embed_chunks(doc_chunks)
    k = min(3, len(doc_chunks))
    top_chunks = top_k(question, doc_embeddings, doc_chunks, k=k)
    answer = generate_answer(top_chunks, question)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "input_text": text,
        "question": question,
        "answer": answer
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

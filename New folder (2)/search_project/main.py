from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from arxiv_search import search_arxiv, save_papers
from summarizer import summarize_papers
from qa_bot import PaperQA
from db import init_db, SessionLocal, Paper
import os
import uvicorn

app = FastAPI()

# Get the absolute base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set paths to static and templates folders using absolute paths
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

# Mount static directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Use absolute path for Jinja2 templates
templates = Jinja2Templates(directory=templates_dir)

# Initialize database and QA bot
init_db()
qa_bot = PaperQA()

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search(request: Request, query: str = Form(...)):
    papers = search_arxiv(query)
    save_papers(papers)
    summarize_papers()

    # Reload QA bot index after papers saved to DB
    try:
        qa_bot.load_papers_from_db()
    except ValueError:
        # Handle case when no abstracts found or DB empty
        pass

    return RedirectResponse(url=f"/results?query={query}", status_code=303)

@app.get("/results")
def results(request: Request, query: str):
    db = SessionLocal()
    papers = db.query(Paper).filter(Paper.title.contains(query)).all()
    db.close()
    return templates.TemplateResponse("results.html", {"request": request, "papers": papers, "query": query})

@app.get("/paper/{paper_id}")
def paper_detail(request: Request, paper_id: int):
    db = SessionLocal()
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    db.close()
    return templates.TemplateResponse("paper_detail.html", {"request": request, "paper": paper, "answer": None})

@app.post("/paper/{paper_id}/qa")
async def paper_qa(request: Request, paper_id: int, question: str = Form(...)):
    try:
        answer = qa_bot.answer_question(question, paper_id)
    except RuntimeError:
        answer = "QA system not ready. Please perform a search first."

    db = SessionLocal()
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    db.close()

    return templates.TemplateResponse("paper_detail.html", {"request": request, "paper": paper, "answer": answer})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

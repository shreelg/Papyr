from db import SessionLocal, Paper
import requests
import xml.etree.ElementTree as ET

def search_arxiv(query, max_results=5, start=0):
    """
    Search arXiv API for papers matching the query.
    
    Args:
        query (str): Search term.
        max_results (int): Number of results to return (max ~30000 per API limits).
        start (int): Offset for pagination.
        
    Returns:
        List[dict]: List of papers with title, abstract, authors.
    """
    url = (
        f'http://export.arxiv.org/api/query?search_query=all:{query}'
        f'&start={start}&max_results={max_results}'
    )
    response = requests.get(url)
    response.raise_for_status()  # Raises error if request failed

    root = ET.fromstring(response.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        abstract = entry.find('atom:summary', ns).text.strip()
        authors = ", ".join(
            author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)
        )
        papers.append({"title": title, "abstract": abstract, "authors": authors})

    return papers

def save_papers(papers):
    """
    Save list of papers to the database if they don't already exist.
    
    Args:
        papers (list of dict): Papers to save.
    """
    db = SessionLocal()
    for p in papers:
        exists = db.query(Paper).filter(Paper.title == p['title']).first()
        if not exists:
            paper = Paper(title=p['title'], abstract=p['abstract'], authors=p['authors'])
            db.add(paper)
    db.commit()
    db.close()

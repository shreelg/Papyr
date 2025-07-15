from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

def generate_answer(context_chunks, question):
    context = " ".join(context_chunks)

    prompt = f"Given the following context and question, provide a clear, concise, and well-structured answer. The answer should be accurate, detailed, and fully address the question, using proper grammar and punctuation. However, do not just copy and paste from the context, rephrase it. Question: {question} Context: {context} Format with proper punctuation and grammar."


    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output_ids = gen_model.generate(inputs["input_ids"], max_length=500)



    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

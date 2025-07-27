from transformers import pipeline
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict
import nltk

nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
llmmodel = genai.GenerativeModel("gemini-2.5-pro")

branches = ["finance", "healthcare", "technology", "education"]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_query(query, branches):
  result = classifier(query,branches)
  return result['labels'][0],result['scores'][0]

#Uploading the data and splitting it into chunks

loader = DirectoryLoader("data/", glob="*.pdf", loader_cls = PyPDFLoader)
documents = loader.load()

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def sentence_token_chunk(text, max_tokens = 400):
  sentences = sent_tokenize(text)
  chunks = []
  current_chunk = []
  current_tokens = 0
  for sentence in sentences:
      sent_tokens = len(tokenizer.encode(sentence, add_special_tokens = False))
      if(current_tokens + sent_tokens > max_tokens) and current_chunk:
        chunks.append(" ".join(current_chunk))
        current_chunk = []
        current_tokens = 0
      current_chunk.append(sentence)
      current_tokens += sent_tokens
  if current_chunk:
     chunks.append(" ".join(current_chunk))
  return chunks  

embeddings = []
payloads = []

for doc in documents:
    text = doc.page_content
    file_name = doc.metadata.get("source","unknown.pdf")
    base_name = os.path.basename(file_name)
    category = base_name.replace('.pdf','').strip().lower()
    for chunk_text in sentence_token_chunk(text, 400):
        emb = model.encode(chunk_text)
        embeddings.append(emb)
        payloads.append({
                    "text": chunk_text,
                    "filename": base_name,
                    "category": category
        })

#start Qdrant as in-memory database
client = QdrantClient(":memory:")

collection_name = "rag_chunks"
embedding_dim = 384

if not client.collection_exists(collection_name=collection_name):
  client.create_collection(
    collection_name = collection_name,
    vectors_config = VectorParams(size=embedding_dim, distance=Distance.COSINE)
  )

#upload embeddings and payloads to Qdrant
client.upload_collection(collection_name = "rag_chunks",
  vectors= embeddings,
  payload=payloads
)

def print_stored_categories(client, collection_name):
    hits = client.scroll(
        collection_name=collection_name,
        with_payload=True,
        limit=1000
    )[0]
    categories = set()
    for hit in hits:
        cat = hit.payload.get("category")
        if cat:
            categories.add(cat)
    print("\n\nCategories stored in Qdrant:", categories)

print_stored_categories(client, collection_name)


'''Retrieve top_k relevant document chunks from Qdrant for a given user query.
    Optionally filter by category (for branched RAG).
    Returns a list of dicts with text, filename, score, etc.'''

def retrieve_relevant_chunks(
  client,
  collection_name: str,
  user_query: str,
  top_k: int = 5,
  category: str = None) -> List[Dict]:

  query_vector = model.encode(user_query)

  query_filter = None
  
  if category:
     category = category.lower()
     print(f"\nFiltering by category: '{category}'\n\n")
     query_filter = Filter(must=[FieldCondition(key = "category", match = MatchValue(value=category))])
  

  search_result = client.search(collection_name=collection_name,
                                  query_vector = query_vector,
                                  limit=top_k,
                                  with_payload=True,
                                  query_filter=query_filter)

  return [{"score" : hit.score,
            "text" : hit.payload.get("text"),
            "filename" : hit.payload.get("filename"),
            "category" : hit.payload.get("category") if "category" in hit.payload else None
          } for hit in search_result]

#Creates summarization pipeline

"""
  Concatenate retrieved chunks and use a summarization model to answer the user's query.
  """
def summarize_chunks(chunks, user_query, max_input_length = 6000):
  
  context = "\n\n".join(chunk["text"] for chunk in chunks[:3] if chunk["text"])

  context = context[:max_input_length]
  
  prompt = f"""
            You are an assistant for question-answering tasks.
            Use the following pieces of context to answer the question as thoroughly and clearly as possible.
            If you don't know the answer, say that you don't know. DON'T MAKE UP ANYTHING.

            {context}

            ---

            Answer the question based on the above context: {user_query}
            """

  summary = llmmodel.generate_content(prompt)
  return summary.text


def main():
  
  user_query = input("Enter your Question: " )

  predicted_branch, confidence = classify_query(user_query,branches)
  print(f"Predicted branch: {predicted_branch} (confidence: {confidence:.2f})")

  #confidence threshold check
  if confidence < 0.4:
        print(f"Warning: I'm not very confident this question belongs to a supported category, The answer might not be accurate.")

  predicted_branch = predicted_branch.strip().lower()
  
  results = retrieve_relevant_chunks(
              client,
              collection_name,
              user_query,
              top_k = 5,
              category=predicted_branch
  )

  if not results:
    print(f"No chunks found for category '{predicted_branch}', searching all.")
    
    results = retrieve_relevant_chunks(
              client,
              collection_name,
              user_query,
              top_k=5,
    )

  final_answer = summarize_chunks(results, user_query)
  print("\n ----- Final Generated Answer --------\n")
  print(final_answer)
  print("\n\n")

if __name__ == "__main__":
    main()


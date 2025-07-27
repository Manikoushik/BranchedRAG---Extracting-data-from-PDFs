# Branched RAG: Retrieval-Augmented Generation with Semantic Branching and Gemini LLM
# 🚀 Overview
Branched RAG is an advanced retrieval-augmented generation (RAG) pipeline that combines semantic query classification (“branching”), modern vector search, and the Gemini 2.5 Pro LLM to provide highly accurate, contextually-relevant answers from large, domain-specific document sets.

This project is a significant upgrade over simple RAG.Each user query is classified into a category (branch), and retrieval happens only within that domain, resulting in faster, more focused answers and true scalability

# ✨ Features
  Gemini 2.5 Pro LLM integration for world-class summarization and QA
  Hybrid chunking (sentence + token based) for meaningful, context-aligned text splits
  Branched retrieval: Each query retrieves only from the most relevant document category
  Qdrant vector database for robust, production-grade vector search
  Dynamic confidence warnings for user trust and transparency
  Easy category debugging and normalization

# 📚 LLM Choice: Why Gemini Over Flan-T5?
What I Used: Gemini 2.5 Pro via Google AI Studio API

Why Not Flan-T5?
  
  I initially tried Google Flan-T5 for summarization and QA, but 
  
  Flan struggled with long prompts: It often produced very short, generic, or copy-paste answers, even when given rich context. 
  Complex, multi-part questions or large context windows often resulted in incomplete or repetitive outputs.

Why Gemini?

  Gemini 2.5 Pro delivered much richer, multi-paragraph, context-aware answers.
  It handled long prompts and nuanced questions far better, supporting a truly production-grade RAG experience.
  Seamless API integration and strong reliability for enterprise use cases.

# ✂️ Chunking: From Sliding Window to Hybrid
Early Approach: Sliding Window
  Used a simple character-based sliding window, which:
    Frequently split sentences and concepts, hurting semantic retrieval.
    Led to awkward, context-poor chunks for LLM input.

Upgraded Approach: Hybrid (Sentence + Token)

  Switched to sentence-based splitting, then grouped sentences to respect a max token limit (using the embedding/generation model’s tokenizer).

  Benefits:

    Preserves natural meaning (no mid-sentence breaks!)
    Fits LLM context windows perfectly
    Boosted both retrieval accuracy and answer quality

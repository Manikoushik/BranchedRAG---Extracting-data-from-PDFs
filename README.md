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
**What I Used:** Gemini 2.5 Pro via Google AI Studio API

**Why Not Flan-T5?**
  
  I initially tried Google Flan-T5 for summarization and QA, but 
  
  Flan struggled with long prompts: It often produced very short, generic, or copy-paste answers, even when given rich context. 
  Complex, multi-part questions or large context windows often resulted in incomplete or repetitive outputs.

**Why Gemini?**

  Gemini 2.5 Pro delivered much richer, multi-paragraph, context-aware answers.
  It handled long prompts and nuanced questions far better, supporting a truly production-grade RAG experience.
  Seamless API integration and strong reliability for enterprise use cases.


# ✂️ Chunking: From Sliding Window to Hybrid
**Early Approach:** Sliding Window
    I initially used a sliding window (character-based chunking) because it’s simple and commonly used.
    However, I found that it often split sentences and concepts awkwardly, leading to poor retrieval and summarization.
    I also wanted to explore chunking methods more suitable for Branched RAG, so I moved to a Hybrid approach—splitting text by sentences and then grouping by token count.


**Upgraded Approach:** Hybrid (Sentence + Token)

  Switched to sentence-based splitting, then grouped sentences to respect a max token limit (using the embedding/generation model’s tokenizer).

  **Benefits:**

  Preserves natural meaning (no mid-sentence breaks!)
  Fits LLM context windows perfectly
  Boosted both retrieval accuracy and answer quality

# 📦 Vector Database: Why Qdrant?
**What I Used:**
  Qdrant (Python in-memory mode for development, scalable for production)

**Why Not Chroma or Milvus?**
  I previously used Chroma for vector search in another project and liked its simplicity for rapid prototyping.For this project, I wanted to try something new and evaluate 
  another leading vector database—Qdrant—for its performance, features, and ease of use in both dev and prod environments.

  Qdrant’s strong metadata filtering, category support, and flexible deployment options made it an ideal fit for Branched RAG.

# 📝 Summarization and Generation
  
  **Prompt engineering is key!** The pipeline builds a prompt using the top retrieved chunks and the user’s question, sent to Gemini for final answer generation.
  Multi-chunk, multi-paragraph, and multi-domain questions are now handled gracefully.

# 🎯 Threshold Confidence
  After query classification, if the confidence is below 0.4, the system issues a user-facing warning:

    "Warning: I'm not very confident this question belongs to a supported category... The answer might not be accurate."

  This helps manage user trust and clarify when answers may be less reliable

  .

# 💡 Challenges & Solutions
**Category Normalization and Matching**
  **Issue:** Early runs produced "No chunks found for category..." because category extraction from filenames (or paths) didn’t always match the classifier’s output (case, whitespace, path errors).

  **Fix:** Used os.path.basename() and .lower().strip() everywhere to normalize both stored and predicted categories.

**Chunking Quality**
  **Issue:** Sliding window chunking led to broken context and poor retrieval.

  **Fix:** Switched to hybrid (sentence + token) chunking using NLTK and the HuggingFace tokenizer.

  **LLM Output Quality**
  **Issue:** Flan-T5 struggled with long prompts and rich context, producing short or unhelpful answers.
  
  **Fix:** Integrated Gemini 2.5 Pro via API—huge improvement for answer detail and accuracy.

**Debugging Every Step**
  Added category printouts, branch diagnostics, and checkpoint logs throughout the code to ensure every stage (classification, chunking, upload, retrieval, summarization) worked as intended.

Certainly! Here’s a rephrased, playful version for your README:

---

## 🛠️ How to Run

If you’ve made it this far, I’m pretty sure you know how to get this project up and running. 😉

# 🧠 Key Takeaways
  **Chunking and category handling are just as important as your LLM choice!**

  **Debug every step as a checkpoint for a truly reliable pipeline.**

  **Branched RAG + Gemini + Qdrant = Next-level retrieval-augmented answers.**



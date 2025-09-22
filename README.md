# csc2701_project

## Problem Statement
RAG (Retrieval Augmented Generation) is one of the many techniques that the recent developments in the area of Natural Language Processing, specifically Larga Language Models, have made possible. Unlike traditionally depending on language models that rely solely on their internal knowledge with which they were trained on, RAG combines the generative properties of LLMs with information retrieval systems to generate more accurate and contextually relevant responses based on real information. This approach addresses the limitations of  knowledge in LLMs by dynamically extracting and incorporating up-to-date or domain-specific information to their context. As a result, RAG has gained attention for applications in open-domain question answering, enterprise information search, healthcare, and more. However, developing and deploying an effective RAG system presents challenges that need to be overcome with effective ML Ops practices.

///
Large Language Models are powerful tools for natural language understanding and generation, but their knowledge is limited to what they were trained on. This makes them prone to outdated or incomplete answers, especially in domains where information changes frequently or is highly specific. Retrieval-Augmented Generation (RAG) addresses this limitation by combining LLMs with information retrieval systems. In RAG pipelines, relevant documents are retrieved from an external knowledge base and injected into the model’s context.

This project applies the RAG approach to answering queries about the MScAC program at University of Toronto. Students often need quick, reliable answers about program policies, courses, and timelines, which are spread across multiple sources. We will be using sources such as the MScAC Student Handbook, the official program website, and course timetables, and by consolidating these sources into a RAG-powered system, we want to provide a better way for students to obtain information.

///

## Data Sources
- What data are we using for the 

## Plan for EDA and modeling

[RAG Systems Techniques List](https://github.com/NirDiamant/RAG_Techniques)


## Initial plan for data exploration and modeling

### Data exploration


<img width="1728" height="1452" alt="image" src="https://github.com/user-attachments/assets/8d0c748f-0800-4632-af58-6cb28d27030a" />


### RAG pipeline 
1. Parsing text data from the data source.
2. Data preprocessing: removing redundant structural emlements like headers/footers, normalizing text 
2. Chunking. Define the chunking strategy - it can be  a fixed-size, a recursive or a context-aware chunking.
3. Building embeddings. Decide between using sparse embeddings for keyword retrieval  (like BM25 or TF-IDF) or dense embeddings (like SBERT) to capture semantic meaning. Or use hybrid search and combine these approaches.
4. Retriever algorithm
5. Reranker algorithm
6. Response generation using top-k chunks and LLM
   
## Considerations for CSC2701

- Simulate changes to the data (such as new data streaming in), and how our pipeline would handle that.

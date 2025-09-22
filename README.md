# csc2701_project

## Problem Statement
RAG (Retrieval Augmented Generation) is one of the many techniques that the recent developments in the area of Natural Language Processing, specifically Larga Language Models, have made possible. Unlike traditionally depending on language models that rely solely on their internal knowledge with which they were trained on, RAG combines the generative properties of LLMs with information retrieval systems to generate more accurate and contextually relevant responses based on real information. This approach addresses the limitations of  knowledge in LLMs by dynamically extracting and incorporating up-to-date or domain-specific information to their context. As a result, RAG has gained attention for applications in open-domain question answering, enterprise information search, healthcare, and more. However, developing and deploying an effective RAG system presents challenges that need to be overcome with effective ML Ops practices.



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

   
### Execution Plan & Consolidation

The primary goal of this project is to demonstrate end-to-end MLOps project on a RAG-based pipeline.  
The machine learning component will be deliberately kept as simple as possible, with most of the effort dedicated to building robust data, deployment, and monitoring workflows.  

---

#### 1. Discovery / Scoping
- **Objective**: Deploy a RAG-based system to answer MScAC program-related queries using handbook and website content.  
- **Scope**:  
  - In-scope:
    - MScAC handbook and MScAC official website as primary knowledge sources.  
    - Fixed-size text chunking and simple embedding-based retrieval.  
    - Vector database for semantic search.  
    - Cloud deployment (AWS) with monitoring.  

  - Out-of-scope:
    - Training/fine-tuning custom LLMs.  
    - Handling multimodal data (e.g., images, tables).  
    - Complex retraining pipelines (only basic re-embedding of updated data). 

---

#### 2. Data Collection  
- Download MScAC handbook (PDF/HTML).  
- Scrape MScAC website and set up automatic update monitoring (AWS Lambda + CloudWatch events?).
- Store all raw sources in AWS S3 with version control.  
- When a change is detected → Lambda fetches updated page, stores it in S3, and triggers embeddings refresh.  


---

#### 3. Preprocessing 
- Normalize text (remove headers, footers, HTML tags).  
- Split into fixed-size chunks.  
- Generate embeddings using a pretrained model (SBERT or OpenAI API).  
- Store vectors in a vector DB (Pinecone / Amazon OpenSearch).

---

#### 4. Experimentation / Validation
- Define a small evaluation set (20–30 representative student queries).  
- Metrics:
  - Retrieval: Recall@k, Precision@k.  
  - ....  
- Baseline: simple vector similarity search (Pinecone / Amazon OpenSearch) + LLM (e.g., OpenAI API). 
---

#### 5. Registration / Governance
- Store raw data, preprocessed chunks, and embeddings in AWS S3 with versioned folders (`/v1`, `/v2`). 
- Use MLflow /logging in GitHub.
- Apply AWS IAM roles to control access to stored data.

---

#### 6. Packaging / Environment
- Containerize the pipeline with Docker.  
- Use FastAPI for serving RAG API.  
- Use Streamlit for a simple client interface.  
- CI/CD with GitHub Actions + AWS CodePipeline? for automated builds and deployments.
- 
---

#### 7. Deployment
- Deploy as an API service on AWS (ECS or Lambda).  
- Endpoints:
  - `/ask` → query + RAG response.  
  - `/update` → trigger re-scraping and re-embedding pipeline.  

---

#### 8. Post-deploy Monitoring
- Track system health: API uptime, latency, and request counts via CloudWatch.  
- Store and inspect RAG logs (retrieved chunks, errors) in CloudWatch Logs.  
- Set CloudWatch Alarms that would trigger SNS notifications
---

#### 9. Feedback & Retrain
- Collect real-world queries and feedback for error analysis.  
- Update embeddings whenever handbook or website content changes.
---

#### 10. Consolidation
- Maintain a living project document with:
  - Data sources and update history.  
  - Embedding model choice and vector DB configuration.  
  - Evaluation results and versioned pipeline.  
- Deliverables:
  - Production-ready RAG API.  
  - Monitoring dashboards in AWS.  


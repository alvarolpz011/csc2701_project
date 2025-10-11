## Problem Statement
RAG (Retrieval Augmented Generation) is one of the many techniques that the recent developments in the area of Natural Language Processing, specifically Larga Language Models, have made possible. Unlike traditionally depending on language models that rely solely on their internal knowledge with which they were trained on, RAG combines the generative properties of LLMs with information retrieval systems to generate more accurate and contextually relevant responses based on real information. This approach addresses the limitations of  knowledge in LLMs by dynamically extracting and incorporating up-to-date or domain-specific information to their context. As a result, RAG has gained attention for applications in open-domain question answering, enterprise information search, healthcare, and more. However, developing and deploying an effective RAG system presents challenges that need to be overcome with effective ML Ops practices.

///
Large Language Models are powerful tools for natural language understanding and generation, but their knowledge is limited to what they were trained on. This makes them prone to outdated or incomplete answers, especially in domains where information changes frequently or is highly specific. Retrieval-Augmented Generation (RAG) addresses this limitation by combining LLMs with information retrieval systems. In RAG pipelines, relevant documents are retrieved from an external knowledge base and injected into the model’s context.

This project applies the RAG approach to answering queries about the MScAC program at University of Toronto. Students often need quick, reliable answers about program policies, courses, and timelines, which are spread across multiple sources. We will be using sources such as the MScAC Student Handbook, the official program website, and course timetables, and by consolidating these sources into a RAG-powered system, we want to provide a better way for students to obtain information.

///




# Evaluation

RAG over unstructured data is a complex system with many components that impact the overall application's quality. Adjusting any single element can have cascading effects on the others. Thus, investing in comprehensive RAG evaluation can pay dividends through improved user satisfaction, reduced operational overhead, and faster iteration cycles.

## 1. Aspects to Evaluate

### 1.1. Evaluating the preprocessing

**Document Extraction Accuracy**
- Correctly extracting content from a wide range of input formats (PDFs, webpages, emails, HTMLs, .docx, etc).
- Especially tricky for:
    - tabular data
    - images with embedded text
    - academic calendars
    - multi-column layouts
    - multi-page content
    - retaining hyperlinks

**Chunking semantic coherence**
- Are documents chunked at appropriate semantic boundaries?
- Can each chunk be understood independently?

**Metadata completeness and accuracy**
- Can the system preserve content / document-level metadata?
- Use attributes like published date, author, etc. to provide enriched answers

### 1.2. Evaluating the retrieval

**Query understanding and expansion**
- Does the system understand the (perhaps unspoken) context behind the user query?

**Search precision and recall**
- What is the precision & recall for extracting the relevant chunks?

**Ranking quality and relevance**
- How well does it rank the chunks in order of question relevance?

### 1.3. Evaluating the generation

**Factual accuracy and faithfulness**
- How much do the RAG’s responses stick exclusively to the context? (i.e. they should not answer from their own knowledge even though it may be correct)
- Could be measured as __# of facts in response derived from context__

**Response completeness**
- Is the returned response sufficient & complete to answer all aspects of the user question?

**Hallucination detection**
- Is the system picking imaginary facts & snippets from the content and providing unfactual responses?


### 1.4. Evaluating the end-to-end system

**User experience quality**
- Is the overall application a smooth experience for users seeking answers?

**Cost-effectiveness**
- Is it cost-effective in terms of infrastructure & LLM costs?

**Scalability and reliability**
- Is it scalable if multiple users are interacting with the app at once?
- Is it reliable at increased loads?


### 1.5. Evaluating the quality in production

Real-World Failure Modes: RAG in production does not work perfectly from the get-go. Even the best models hallucinate when evaluated for hallucination index.

**Critical Scenarios to Test:**

**1. Hallucination Scenarios**
- Information not in context documents
- Mixing facts from different sources incorrectly
- Generating plausible but false information

**2. Noise Robustness**
- Extracting relevant info from mixed relevant/irrelevant context
- Handling contradictory information across documents
- Managing information overload

**3. Negative Rejection**
- Declining to answer when insufficient context
- Recognizing out-of-scope queries
- Appropriate uncertainty expression

**4. Privacy & Security**
- PII leak prevention
- Sensitive information handling
- Malicious query detection

**5. Domain Adherence**
- Staying within intended use case boundaries
- Handling off-topic queries appropriately
- Maintaining role consistency


# 2. Technical Framework for Evaluation

### 2.1. Establish the Metrics

#### 2.1.1. For Retrieval

**Classical Information Retrieval Metrics**
- Precision@k = |Relevant ∩ Retrieved@k| / k
- Recall@k = |Relevant ∩ Retrieved@k| / |Relevant|
- F1@k = 2 × (Precision@k × Recall@k) / (Precision@k + Recall@k)

- Mean Reciprocal Rank (MRR) = (1/|Q|) × Σ(1/rank_i)
- Normalized DCG@k = DCG@k / IDCG@k

**RAG-Specific Metrics**
- **Context Relevancy**: How relevant retrieved context is to the query
- **Context Recall**: Whether retrieval contains all necessary information
- **Context Precision**: Whether retrieved context is ranked correctly
- **Hit Rate@k**: Percentage of queries with at least one relevant result in top-k

RAG evaluations often adapt precision and recall to work on statements instead of documents, calling a retrieved statement "relevant" if it was present in the ground-truth context.

#### 2.1.2. For Generation

**Core Generation Metrics**
- **Answer Relevancy**: Relevance of generated response to query
- **Faithfulness**: Whether response is grounded in retrieved context
- **Completeness**: How well response incorporates all relevant context
- **Consistency**: Response stability across similar queries

**Advanced Quality Metrics**
- **Attribution**: Ability to trace claims to source documents
- **Hallucination Detection**: Identification of fabricated information
- **Refusal Quality**: Appropriate handling of unanswerable queries
- **Tone Adherence**: Maintenance of desired communication style

### 2.2. Use LLM-as-a-Judge

A modern and increasingly popular approach for evaluating RAG outputs is the use of large language models as evaluators, commonly referred to as "LLM-as-a-Judge".

**Pseudocode of Implementation**
```python
def evaluate_faithfulness(query, context, response, judge_model):
    prompt = f"""
    Evaluate if the response is factually grounded in the context.
    
    Query: {query}
    Context: {context}
    Response: {response}
    
    Rate faithfulness on 1-5 scale:
    1: Completely unfaithful, major hallucinations
    5: Completely faithful, all claims supported
    
    Provide reasoning then score.
    """
    return judge_model.generate(prompt)
```

**Best Practices for LLM Judges:**
- Use detailed scoring rubrics (1-5 scales with explicit criteria)
- Require reasoning before scoring for consistency
- Validate judge reliability against human annotations
- Use different models for different evaluation aspects
- Implement cross-validation across multiple judges

#### 2.2.1. Use Synthetic Data

- We can speed things up by generating synthetic test cases directly from our knowledge base by flipping the usual RAG workflow
- Start from content and ask an LLM to generate questions and answers

## 3. Explore Libraries & Frameworks

**RAGAS** - Most Popular
- Comprehensive metric suite
- Easy integration with popular frameworks
- Strong community support
- Limitations: Academic focus, limited customization

**DeepEval** - Production-Oriented
- Unit testing approach for LLMs
- 40+ vulnerability tests
- CI/CD integration
- Real-time monitoring capabilities

**Arize Phoenix** - Observability Focus
- Step-by-step trace visualization
- Real-time monitoring
- Custom evaluator support
- Open-source with enterprise features

**Galileo** - End-to-End Platform
- Unified workflow management
- Proprietary evaluation metrics
- Production-scale observability
- SOC 2 compliance and security

**LangSmith** - Developer-Friendly
- Comprehensive logging and tracing
- A/B testing capabilities
- Multi-provider support
- Strong debugging tools
- 

## Data Sources & Extraction
Given the nature of RAG systems, data sources can vary depending on the applications. Commonly, data sources can include:
* Text documents: Articles, reports, books, etc. Common formats can include PDF, DOCX, TXT, HTML, etc.
* Databases: Structured data from SQL or NoSQL databases.
* APIs: Data from web services or other online sources.
* Multimedia, such as images, audio, and video.
* Other knowledge bases like webpages, wikis, etc.

For this specific project, we will be using a collection of informational PDF documents aimed at MSCAC students. These documents include:
* MSCAC 2025-2026 Student Handbook
* The MScAC [Program Webpage](https://mscac.utoronto.ca/)
* The CS 2025-2026 Fall/Winter [Graduate Course Timetables Webpage ](https://web.cs.toronto.edu/graduate/timetable)

In all of these sources, the data must be cleaned and preprocessed to ensure that it is in a suitable format for the RAG system to produce accurate and relevant responses. In the case of the PDF documents, text extraction techniques will be needed to convert its content into structured text data easily understandable by the RAG system generativve model. As for the webpages, web scraping techniques will be needed to extract the desired information from the HTML pages, which could be found as plain text, tables, lists, images, etc.

The data extraction process will need a defined pipeline to deal with these data formats. The following steps will be taken:
1. **Data Collection**: Gather all relevant documents and webpages via an automated web scraper or manual download.
2. **Text Extraction**: Use specialized libraries (PyMuPDF, PyPDF2, pdfplumber, etc.) to extract text from PDF documents. For webpages, options include BeautifulSoup, Scrapy, Selenium, or even AI-based web scrappers.
3. **Data Cleaning**: Remove any unnecessary elements, normalize the text by converting it to lowercase and removing extra spaces.
4. **Text Chunking**: Split the text into smaller, manageable chunks for processing before being saved into a database or vector store.
5. **Data Storage**: Store the cleaned and chunked text data in a suitable format, such as a database or a vector store, for easy retrieval during the RAG process.

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




## FastApi 

### 1. Run the app locally:

 uvicorn main:app --host 0.0.0.0 --port 8080

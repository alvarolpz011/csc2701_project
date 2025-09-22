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

## Evaluation

RAG over unstructured data is a complex system with many components that impact the overall application's quality. Adjusting any single element can have cascading effects on the others. Thus, investing in comprehensive RAG evaluation can pay dividends through improved user satisfaction, reduced operational overhead, and faster iteration cycles.

### Aspects to Evaluate

#### Evaluating the preprocessing

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

#### Evaluating the retrieval

**Query understanding and expansion**
- Does the system understand the (perhaps unspoken) context behind the user query?

**Search precision and recall**
- What is the precision & recall for extracting the relevant chunks?

**Ranking quality and relevance**
- How well does it rank the chunks in order of question relevance?

#### Evaluating the generation

**Factual accuracy and faithfulness**
- How much do the RAG’s responses stick exclusively to the context? (i.e. they should not answer from their own knowledge even though it may be correct)
- Could be measured as __# of facts in response derived from context__

**Response completeness**
- Is the returned response sufficient & complete to answer all aspects of the user question?

**Hallucination detection**
- Is the system picking imaginary facts & snippets from the content and providing unfactual responses?


#### Evaluating the end-to-end system

**User experience quality**
- Is the overall application a smooth experience for users seeking answers?

**Cost-effectiveness**
- Is it cost-effective in terms of infrastructure & LLM costs?

**Scalability and reliability**
- Is it scalable if multiple users are interacting with the app at once?
- Is it reliable at increased loads?


#### Evaluating the quality in production

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


### Technical Framework for Evaluation

#### Establish the Metrics

##### For Retrieval

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

##### For Generation

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

#### Use LLM-as-a-Judge

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

#### Use Synthetic Data

- We can speed things up by generating synthetic test cases directly from our knowledge base by flipping the usual RAG workflow
- Start from content and ask an LLM to generate questions and answers

### Explore Libraries & Frameworks

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


## Considerations for CSC2701

- Simulate changes to the data (such as new data streaming in), and how our pipeline would handle that.

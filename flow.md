# Flow of a Request

## Frontend: Streamlit (runs on EC2 instance: `2701_backup`)
1. User enters query (along with conversation history)
2. POST request with user_query to FastAPI backend endpoint at API_URL ('localhost:8080/chat')
3. Display response

## Backend: FastAPI (runs on EC2 instance: `2701_backup`)
4. Accepts request at port 8080
5. Pre-processes user query
6. Passes to `src.rag_architecture.rag`
7. Returns reponse

## RAG Code: Python (runs on EC2 instance: `2701_backup`)
8. Augment user query
9. Embed user query (using SentenceTransformers - locally, not API)
10. Retrieve top-k relevant chunks
    a. Initialize QdrantClient('<our_qdrant_url>') (runs on EC2 instance: `2701_vector_db_instance`)
    b. Take our embedded User Query `self.query_vector`
    c. Ask for top k chunks (using dense vectors - not sparse, hybrid)
11. Make prompt
12. Ask LLM
13. Return LLM response

## Deployment Stuff
1. Backend runs within a Docker container

1. Tag & build Docker image from Dockerfile locally: takes ~5 mins
2. Push our Docker image to AWS Elastic Container Registry (ECR): takes ~30 mins
3. Pull our image into `2701_backup`: takes ~2/3 mins
4. Map the ports & run our Docker image: instantenous
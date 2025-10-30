"""
RAG (Retrieval-Augmented Generation) implementation for MScAC handbook Q&A.

This module provides classes for:
- Similarity measurement between embeddings
- RAG pipeline for question answering over document chunks
"""

import torch
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from .utils import TEMP_CHUNKS, ask_llm, genai, types, load_dotenv, os, base64
from .prompts import RETRIEVAL_PROMPT
from dotenv import load_dotenv
import json
load_dotenv()


class SimilarityMeasure:
    """
    Computes similarity between two vectors using various metrics.

    Args:
        X: First vector (query embedding)
        Y: Second vector (document embedding)
        metric_type: Type of similarity metric ('cosine_similarity', 'dot_product', or 'euclidean_distance')

    Attributes:
        similarity: Computed similarity score
    """

    def __init__(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            metric_type: str = 'cosine_similarity'
        ):

        self.X, self.Y = X, Y
        self.metric_type = metric_type
        self.similarity = None

        if self.metric_type == 'cosine_similarity':
            self.similarity = self.cosine_similarity()

        elif self.metric_type == 'dot_product':
            self.similarity = self.dot_product()

        elif self.metric_type == 'euclidean_distance':
            self.similarity = self.euclidean_distance()

    def cosine_similarity(self):
        """Compute cosine similarity between X and Y."""
        return np.dot(self.X, self.Y) / (np.linalg.norm(self.X) * np.linalg.norm(self.Y))

    def dot_product(self):
        """Compute dot product between X and Y."""
        return np.dot(self.X, self.Y)

    def euclidean_distance(self):
        """Compute negative euclidean distance (higher = more similar)."""
        return -np.linalg.norm(self.X - self.Y)


class RAG:
    """
    Retrieval-Augmented Generation pipeline for question answering.

    Workflow:
    1. Embed user query using sentence transformer
    2. Retrieve top-k most similar chunks from vector database
    3. Prepare prompt with retrieved context
    4. Generate response using language model

    Args:
        vector_db_url: URL of the vector database (currently unused, placeholder)
        embedding_model_name: Name of the SentenceTransformer model
        language_model_name: Name of the language model for generation
    """

    def __init__(
            self,
            vector_db_url: str = 'http://172.31.41.249:6333',
            vector_db_collection: str= "csc2701",
            embedding_model_name: str = 'all-MiniLM-L6-v2',
            language_model_name: str = 'gemini-2.5-flash-lite'
        ):

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.vector_db_url = vector_db_url
        self.vector_db_collection = vector_db_collection
        self.vector_db_client = QdrantClient(self.vector_db_url)
        self.chunks = self.load_chunks_from_vector_db(self.vector_db_url)
        self.language_model = language_model_name 

    def load_chunks_from_vector_db(self, vector_db_url: str):
        """
        Load and embed document chunks from vector database.

        Currently uses TEMP_CHUNKS from utils as a placeholder.

        Returns:
            List of tuples (chunk_text, chunk_embedding)
        """
        chunks = []
        for chunk_text in TEMP_CHUNKS:
            chunk_embedding = self.embedding_model.encode(chunk_text)
            chunks.append((chunk_text, chunk_embedding))

        return chunks
    def augment_user_query(self, user_query: str, num_questions: int = 3) -> str:
        """
        Augment the user's query by generating related questions
        using the Gemini API. These related questions help provide
        richer context for retrieval.
        """
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        model = self.language_model

        prompt = f'''Given this user query: "{user_query}"

        Generate exactly {num_questions} related questions that someone asking this might also want to know.
        Some background info about the program:
        The MScAC program is a 16-month applied research program designed to educate the next generation of world-class innovators. Students enrol in advanced graduate courses according to the concentration requirements. They also complete an eight-month applied research internship, usually paid, based at an industry partner.
        Rules:
        - Make questions specific and directly related to the original query
        - Cover different aspects (requirements, deadlines, process, eligibility, etc.)
        - Keep questions concise and clear
        - Return ONLY a JSON array of questions, nothing else
        
        Example format: ["question1?", "question2?", "question3?"]
        '''

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.7,
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        response_text = response.text.strip()

        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        questions = json.loads(response_text)
        questions_str = " ".join(questions)
        augmented_query = f"{user_query} {questions_str}"

        return augmented_query

    def embed_user_query(self):
        """Embed the user query using the sentence transformer model."""
        return self.embedding_model.encode(self.user_query)

    def retrieve_top_k_relevant_chunks(self, top_k: int = 3):
        """
        Retrieve top-k most similar chunks to the query.

        Args:
            top_k: Number of chunks to retrieve

        Returns:
            List of tuples (chunk_text, similarity_score) sorted by similarity
        """
        similarities = []
        
        search_results = self.vector_db_client.search(
            collection_name=self.vector_db_collection,
            query_vector= models.NamedVector(name= "mscac-dense-vector", vector= self.query_vector),
            limit=top_k,
        )
        print(search_results)
        for result in search_results:
            similarities.append((result.payload["header"], result.payload["document_title"], result.payload["content"], result.score)) 
            #print(result.payload["header"], result.score)
                
        #for chunk, chunk_embedding in self.chunks:
        #    similarity = SimilarityMeasure(self.query_vector, chunk_embedding).similarity
        #    similarities.append((chunk, similarity))

        similarities.sort(key=lambda x: x[3], reverse=True)
        return similarities

    def preprocess_chunk(self, chunk: str) -> str:
        """
        Preprocess a single chunk by:
        - Removing excessive whitespace and newlines
        - Removing special characters and control characters
        - Normalizing spacing

        Args:
            chunk: Raw chunk text

        Returns:
            Cleaned chunk text
        """
        if not chunk or not isinstance(chunk, str):
            return ""

        # Remove control characters and non-printable characters
        chunk = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', chunk)

        # Replace multiple newlines with a single space
        chunk = re.sub(r'\n+', ' ', chunk)

        # Replace multiple spaces/tabs with a single space
        chunk = re.sub(r'\s+', ' ', chunk)

        # Remove leading/trailing whitespace
        chunk = chunk.strip()

        return chunk
    
    def rag_prompt(self) -> str:
        context_section = ""
        for i, chunk_tuple in enumerate(self.top_k_chunks):
            chunk_text = chunk_tuple[2]  # NOT USING TEXT DOCUMENT OR TITLE YET
            context_section += f"[Chunk {i}]\n{chunk_text}\n\n"

        prompt = f"""You are a helpful assistant answering questions based on the provided context.
        
        CONTEXT:
        {context_section}
        
        USER QUESTION:
        {self.user_query}
        
        INSTRUCTIONS:
        - Answer the question using ONLY the information provided in the context above
        - If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question fully."
        - Be specific and detailed in your answer
        - Use a natural, conversational tone
        
        ANSWER:"""
        return prompt
    '''
    def query_llm(self) -> str:
        prompt = self.rag_prompt()
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        model = self.language_model

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.3,
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        return response.text

    def generate_response_to_user(self):
        return self.query_llm()
        
    def prepare_retrieval_prompt(self):
        """
        Prepare the retrieval prompt by:
        - Preprocessing each chunk to remove extraneous characters
        - Formatting chunks with clear delimiters
        - Creating a well-structured context string

        Returns:
            Formatted prompt for the language model
        """
        if not self.top_k_chunks:
            context = "No relevant context found."
        else:
            # Preprocess each chunk (extract text from tuple if needed)
            preprocessed_chunks = []
            for item in self.top_k_chunks:
                # Handle (chunk_text, similarity_score) tuple format
                chunk_text = item[0] if isinstance(item, tuple) else item
                processed = self.preprocess_chunk(chunk_text)
                if processed:
                    preprocessed_chunks.append(processed)

            # Format chunks with numbering for clarity
            if preprocessed_chunks:
                context = "\n\n".join([
                    f"[{i+1}] {chunk}"
                    for i, chunk in enumerate(preprocessed_chunks)
                ])
            else:
                context = "No relevant context found."

        return RETRIEVAL_PROMPT.format(
            context=context,
            question=self.user_query
        )

    def generate_response_to_user(self):
        """Generate a response using the language model."""
        return utils.ask_llm(
            user_prompt=self.retrieval_prompt,
            model=self.language_model
        )
    '''

    def __call__(self, user_query: str, top_k: int = 3):
        """
        Execute the RAG pipeline.

        Args:
            user_query: User's question
            top_k: Number of chunks to retrieve

        Returns:
            Generated answer from the language model
        """
        self.user_query = self.augment_user_query(user_query)
        self.query_vector = self.embed_user_query()
        self.top_k_chunks = self.retrieve_top_k_relevant_chunks(top_k)
        self.retrieval_prompt = self.rag_prompt()
        self.response_to_user = ask_llm(user_prompt=self.retrieval_prompt,model=self.language_model)
        
        return self.response_to_user

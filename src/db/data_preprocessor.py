import re
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


class DataPreprocessor:
    def __init__(self, file_path, model_name="all-MiniLM-L6-v2", vector_db_url="http://localhost:6333"):
        self.file_path = file_path
        self.vector_db_url = vector_db_url
        self.model = SentenceTransformer(model_name)

        self.hand_book_txt = self.read_text_file()
        self.hand_book_text_chunks = self.split_by_headers()
        self.hand_book_embeddings = self.create_embeddings()
        self.upload_to_vector_db()

    def read_text_file(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            return file.read()

    def split_by_headers(self) -> List[Dict[str, str]]:

        text = self.hand_book_txt.replace('\r', '')

        header_pattern = re.compile(
            r'(?P<header>^[A-Z0-9 ,\-&/\(\)]+(?:\n|$))', re.MULTILINE
        )

        parts = re.split(header_pattern, text)

        chunks = []
        current_header = None

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.isupper() or re.match(r'^[A-Z0-9 ,\-&/\(\)]+$', part):
                current_header = part
            else:
                if current_header:
                    clean_text = re.sub(r'\n{2,}', '\n', part).strip()
                    chunks.append({
                        "title": current_header,
                        "content": clean_text
                    })
                    current_header = None
                else:
                    chunks.append({
                        "title": "PREFACE",
                        "content": part
                    })
        return chunks

    def create_embeddings(self):
        embeddings_dict = {}
        for d in self.hand_book_text_chunks:
            embeddings_dict[d["title"]] = self.model.encode(d['content'])
        return embeddings_dict

    def upload_to_vector_db(self):
        client = QdrantClient(self.vector_db_url)
        client.upsert(
            collection_name="csc2701",
            points=[
                PointStruct(
                    id=idx,
                    vector={"mscac-dense-vector": vector.tolist()},
                    payload={
                        "header": title,
                    }
                )
                for idx, (title, vector) in enumerate(self.hand_book_embeddings.items())
            ]
        )


if __name__ == "__main__":
    hand_book_path = "../data/docs/handbook.txt"
    data_preprocessor = DataPreprocessor(hand_book_path)

import time
from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DB_PATH = Path(__file__).resolve().parent.parent / "memory_db"


class MemoryStore:
    def __init__(self, path: str = str(DB_PATH)):
        self.client = PersistentClient(path=path)
        self.ef = DefaultEmbeddingFunction()
        self.semantic = self.client.get_or_create_collection(
            name="semantic", embedding_function=self.ef
        )
        self.episodic = self.client.get_or_create_collection(
            name="episodic", embedding_function=self.ef
        )

    def add(self, collection: str, content: str, metadata: dict | None = None):
        col = self.semantic if collection == "semantic" else self.episodic
        col.add(
            ids=[f"{collection}_{int(time.time() * 1000)}_{id(content)}"],
            documents=[content],
            metadatas=[metadata or {}]
        )

    def query(self, collection: str, text: str, n: int = 5) -> list[str]:
        col = self.semantic if collection == "semantic" else self.episodic
        count = col.count()
        if count == 0:
            return []
        n_results = min(n, count)
        results = col.query(query_texts=[text], n_results=n_results)
        if results and results.get("documents") and results["documents"][0]:
            return results["documents"][0]
        return []

    def count(self, collection: str) -> int:
        col = self.semantic if collection == "semantic" else self.episodic
        return col.count()


memory_store = MemoryStore()

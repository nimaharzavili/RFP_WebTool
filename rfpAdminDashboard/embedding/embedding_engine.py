import nest_asyncio

nest_asyncio.apply()
import chromadb
from llama_parse import LlamaParse
from copy import deepcopy
import os, pickle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
from dotenv import load_dotenv
load_dotenv()
# node_parser = MarkdownElementNodeParser(num_workers=8)

# reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

#db = chromadb.PersistentClient(path=f"{os.path.join('rfpApp','vector_database')}")

class DocumentAnalysis:
    def __init__(self, useNode=True, specificTypesList=None):
        self.useNode = useNode
        self.specificTypesList = specificTypesList

    def setupParser(self):
        parser = LlamaParse(
            api_key= os.getenv('LLAMA_CLOUD_KEY'),  
            result_type="markdown",  
            num_workers=8,   
            verbose=True,
            language="en",  
        )
        return parser
    
    def loadDocuments(self):
        parser = self.setupParser()
        path_ = os.path.join('rfpAdminDashboard', 'data', 'RFP_files')

        all_supported_extensions = [".pdf", ".docx", ".pptx", ".xlsx"]  
        if self.specificTypesList:
            supported_extensions = [ext for ext in self.specificTypesList if ext in all_supported_extensions]
        else:
            supported_extensions = all_supported_extensions
        print(supported_extensions)
        file_extractor = {ext: parser for ext in supported_extensions}
        documents = SimpleDirectoryReader(
            path_, file_extractor=file_extractor
        ).load_data()
        return documents

    def getPageNodes(self, docs, separator="\n---\n"):
        """Split each document into page node, by separator."""
        nodes = []
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                node = TextNode(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                nodes.append(node)
        return nodes
    
    def createVectorStore(self):
        db = chromadb.PersistentClient(path='./ChromaDb')
        chroma_collection = db.create_collection("rfpVectorDatabase6")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context
    
    def indexing(self):
        index_file_path = os.path.join('static', 'index.pkl')
        documents = self.loadDocuments()
        if self.useNode:
            page_nodes = self.getPageNodes(documents)
            index = VectorStoreIndex(nodes=page_nodes)
        else:
            index = VectorStoreIndex.from_documents(documents)

        with open(index_file_path, 'wb') as index_file:
            pickle.dump(index, index_file)

import nest_asyncio

nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
import os, time
from llama_index.core.postprocessor import SentenceTransformerRerank
node_parser = MarkdownElementNodeParser(num_workers=8)

reranker = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
os.environ["OPENAI_API_KEY"] = ""

class DocumentParser:
    def __init__(self, document_type='pdf'):
        self.document_type = document_type

    def setupParser(self):
        parser = LlamaParse(
            api_key= "",  
            result_type="markdown",  
            num_workers=4,   
            verbose=True,
            language="en",  
        )
        return parser
    
    def loadDocuments(self):
        parser = self.setupParser()
        path_ = os.path.join('rfpApp','static', 'RFP_samples')
        if self.document_type == 'pdf':
            # if target documents are pdf, then filter docx files
            file_extractor = {".docx": parser}
        else:
            file_extractor = {".pdf": parser}
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
    
    def documentParserDriver(self):
        documents = self.loadDocuments()
        page_nodes = self.getPageNodes(documents)
        print(page_nodes[5].get_content())

    def flat_indexing(self, query_text):
        path_ = os.path.join('rfpApp','static', 'RFP_samples', 'RFP2023-006060 Service Centre Services_Ontario Health.pdf')
        reader = SimpleDirectoryReader(input_files=[path_])
        base_docs = reader.load_data()
        
        start_time = time.time()
        raw_index = VectorStoreIndex.from_documents(base_docs)
        exec_time = time.time() - start_time
        print(f"Total indexing time: {exec_time}")

        raw_query_engine = raw_index.as_query_engine(
            similarity_top_k=5, node_postprocessors=[reranker]
        )
        start_time = time.time()
        response = raw_query_engine.query(query_text)
        exec_time = time.time() - start_time

        print(f"Total qeury time: {exec_time}")
        return response

    def recursive_indexing(self, query_text):
        start_time = time.time()
        documents = self.loadDocuments()
        exec_time = time.time() - start_time
        print(f"Total document loading time: {exec_time}")

        # start_time = time.time()
        # nodes = node_parser.get_nodes_from_documents(documents)
        # exec_time = time.time() - start_time
        # print(f"Total document parsing time: {exec_time}")

        # base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        page_nodes = self.getPageNodes(documents)

        start_time = time.time()
        recursive_index = VectorStoreIndex(nodes=page_nodes)
        exec_time = time.time() - start_time
        print(f"Total indexing time: {exec_time}")

        recursive_query_engine = recursive_index.as_query_engine(
            similarity_top_k=5, node_postprocessors=[reranker], verbose=True
        )

        start_time = time.time()
        response = recursive_query_engine.query(query_text)
        exec_time = time.time() - start_time
        print(f"Total qeury time: {exec_time}")
        return response
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

class KnowledgeBase:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vector_store = None
        self.docs_dir = 'knowledge_docs'
        
        # 确保文档目录存在
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir)
    
    def load_documents(self):
        """加载知识库目录中的所有文档"""
        documents = []
        for file in os.listdir(self.docs_dir):
            if file.endswith(('.txt', '.md', '.pdf')):
                file_path = os.path.join(self.docs_dir, file)
                loader = UnstructuredFileLoader(file_path)
                documents.extend(loader.load())
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        return split_docs
    
    def initialize_vector_store(self):
        """初始化向量数据库"""
        documents = self.load_documents()
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            return True
        return False
    
    def search_knowledge(self, query, k=3):
        """搜索相关知识"""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    
    def add_document(self, file_path):
        """添加新文档到知识库"""
        if not os.path.exists(file_path):
            return False
        
        # 复制文件到知识库目录
        filename = os.path.basename(file_path)
        target_path = os.path.join(self.docs_dir, filename)
        with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
            dst.write(src.read())
        
        # 重新初始化向量数据库
        self.initialize_vector_store()
        return True
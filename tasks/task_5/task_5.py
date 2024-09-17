import sys
import os
import streamlit as st

sys.path.append(os.path.abspath('../../'))

from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

class ChromaCollectionCreator:
    def __init__(self, processor, embed_model):
        """
        Initializes the ChromaCollectionCreator with a DocumentProcessor instance and embeddings configuration.
        :param processor: An instance of DocumentProcessor that has processed documents.
        :param embed_model: An embedding client for embedding documents.
        """
        self.processor = processor     
        self.embed_model = embed_model  
        self.db = None                  
    
    def create_chroma_collection(self):
        """
        Task: Create a Chroma collection from the documents processed by the DocumentProcessor instance.
        """
        
        if len(self.processor.pages) == 0:
            st.error("No documents found!", icon="🚨")
            return
           
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=50)
        texts = []
        for page in self.processor.pages:
            texts.extend(splitter.split_text(page))

        if texts:
            st.success(f"Successfully split pages to {len(texts)} documents!", icon="✅")

        
        self.db = Chroma.from_documents(
            documents=[Document(text=text) for text in texts],
            embedding=self.embed_model
        )

        if self.db:
            st.success("Successfully created Chroma Collection!", icon="✅")
        else:
            st.error("Failed to create Chroma Collection!", icon="🚨")
    
    def query_chroma_collection(self, query) -> Document:
        """
        Queries the created Chroma collection for documents similar to the query.
        :param query: The query string to search for in the Chroma collection.
        
        Returns the first matching document from the collection with similarity score.
        """
        if self.db:
            docs = self.db.similarity_search_with_relevance_scores(query)
            if docs:
                return docs[0]
            else:
                st.error("No matching documents found!", icon="🚨")
        else:
            st.error("Chroma Collection has not been created!", icon="🚨")

if __name__ == "__main__":
    processor = DocumentProcessor() 
    processor.ingest_documents()
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "YOUR PROJECT ID HERE",
        "location": "us-central1"
    }
    
    embed_client = EmbeddingClient(**embed_config)  
    
    chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
    with st.form("Load Data to Chroma"):
        st.write("Select PDFs for Ingestion, then click Submit")
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            chroma_creator.create_chroma_collection()
import sys
import os
import streamlit as st
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator

if __name__ == "__main__":
    st.header("Quizzify")

    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "YOUR PROJECT ID HERE",
        "location": "us-central1"
    }

    screen = st.empty()
    with screen.container():
        st.header("Quizzify")

        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)

        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

            topic_input = st.text_input("Enter the quiz topic:")
            
            num_questions = st.slider("Select the number of questions:", min_value=1, max_value=20, value=5)

            document = None
            
            submitted = st.form_submit_button("Generate a Quiz!")
            if submitted:

                chroma_creator.create_chroma_collection()

                if chroma_creator.db:
                    st.success("Chroma collection created successfully!")
                    
                    document = chroma_creator.query_chroma_collection(topic_input)
                else:
                    st.error("Failed to create Chroma collection. Please try again.")

    if document:
        screen.empty()
        with st.container():
            st.header("Query Chroma for Topic, Top Document: ")
            st.write(document)
import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator
from tasks.task_8.task_8 import QuizGenerator

class QuizManager:
    ##########################################################
    def __init__(self, questions: list):
        """
        Initializes the QuizManager class with a list of quiz questions.

        :param questions: A list of dictionaries, where each dictionary represents a quiz question along with its choices, correct answer, and an explanation.
        """
        # Store the provided list of quiz question objects
        self.questions = questions
        # Calculate and store the total number of questions
        self.total_questions = len(questions)
    ##########################################################

    def get_question_at_index(self, index: int):
        """
        Retrieves the quiz question object at the specified index. If the index is out of bounds,
        it restarts from the beginning index.

        :param index: The index of the question to retrieve.
        :return: The quiz question object at the specified index, with indexing wrapping around if out of bounds.
        """
        # Ensure index is always within bounds using modulo arithmetic
        valid_index = index % self.total_questions
        return self.questions[valid_index]
    
    ##########################################################
    def next_question_index(self, direction=1):
        """
        Adjusts the current quiz question index based on the specified direction.

        :param direction: An integer indicating the direction to move in the quiz questions list (1 for next, -1 for previous).
        """
        # Retrieve the current question index from Streamlit's session state
        current_index = st.session_state.get("question_index", 0)
        # Adjust the index based on the provided direction using modulo arithmetic
        new_index = (current_index + direction) % self.total_questions
        # Update the question_index in Streamlit's session state with the new, valid index
        st.session_state["question_index"] = new_index
    ##########################################################


# Test Generating the Quiz
if __name__ == "__main__":
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "YOUR-PROJECT-ID-HERE",
        "location": "us-central1"
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()
    
        embed_client = EmbeddingClient(**embed_config) 
    
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
        question = None
        question_bank = None
    
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                
                st.write(topic_input)
                
                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question_bank = generator.generate_quiz()

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            
            # Task 9
            ##########################################################
            quiz_manager = QuizManager(question_bank)  # Use our new QuizManager class
            # Format the question and display
            with st.form("Multiple Choice Question"):
                ##### YOUR CODE HERE #####
                index_question = quiz_manager.get_question_at_index(st.session_state.get("question_index", 0))
                ##### YOUR CODE HERE #####
                
                # Unpack choices for radio
                choices = []
                for choice in index_question['choices']:  # For loop unpack the data structure
                    ##### YOUR CODE HERE #####
                    key = choice['key']  # Set the key from the index question 
                    value = choice['value']  # Set the value from the index question
                    ##### YOUR CODE HERE #####
                    choices.append(f"{key}) {value}")
                
                ##### YOUR CODE HERE #####
                # Display the question onto streamlit
                st.write(index_question["question"])
                ##### YOUR CODE HERE #####
                
                answer = st.radio(  # Display the radio button with the choices
                    'Choose the correct answer',
                    choices
                )
                submitted_answer = st.form_submit_button("Submit")
                
                if submitted_answer:  # On click submit 
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):  # Check if answer is correct
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
            ##########################################################

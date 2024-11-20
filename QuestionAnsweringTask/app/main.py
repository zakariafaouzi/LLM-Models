"""La fonction main"""

import streamlit as st
from QuestionAnswerApp import DataLoader
from QuestionAnswerApp import EmbeddingsManager
from QuestionAnswerApp import QuestionAnswerSystem


# Chemins des fichiers de données
paths = [r"C:\Users\Zakaria-Laptop\LLM-Models\QuestionAnsweringTask\DataBase\S08_question_answer_pairs.txt",
         r"C:\Users\Zakaria-Laptop\LLM-Models\QuestionAnsweringTask\DataBase\S09_question_answer_pairs.txt",
         r"C:\Users\Zakaria-Laptop\LLM-Models\QuestionAnsweringTask\DataBase\S10_question_answer_pairs.txt"]


# Chargement des données et préparation des questions et réponses
data_loader = DataLoader(paths)
data_loader.lire_data()
data_loader.prepare_dataset()
questions = data_loader.data['Question'].tolist()
responses = data_loader.data['Answer'].tolist()

# Initialiser EmbeddingsManager et charger les embeddings sauvegardés
embeddings_manager = EmbeddingsManager()
embeddings_manager.load_embedding()

# Initialiser le système de question-réponse
qa_system = QuestionAnswerSystem(embeddings_manager, questions, responses)


# --- Interface Streamlit ---

st.title("Système de Question-Réponse avec BERT Embeddings")
st.write("Posez une question et le modèle tentera de trouver la réponse la plus appropriée en utilisant les embeddings BERT.")

# Champ de texte pour poser la question
new_question = st.text_input("Votre question:")

# Bouton pour lancer la recherche de réponse
if st.button("Trouver la réponse"):
    if new_question:
        response_bert = qa_system.find_answer(new_question)
        st.write(f"**Question:** {new_question}")
        st.write(f"**Réponse:** {response_bert}")
    else:
        st.write("Veuillez entrer une question.")

# Option pour afficher les questions/réponses disponibles dans le jeu de données
if st.checkbox("Afficher les questions disponibles dans le jeu de données"):
    st.write(questions)

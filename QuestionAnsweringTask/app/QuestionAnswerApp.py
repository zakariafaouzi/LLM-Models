"""La definition des classes"""
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset, DatasetDict

class DataLoader:

    """ Classe pour préparer la base de données"""

    def __init__(self, paths):
        self.paths = paths
        self.data = None
        self.dataset_dict = None   
    
    def lire_data(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """        """_summary_

        Args:
            self (Self): _description_

        Returns:
            pd.DataFrame: _description_
        """
        dataframes = [pd.read_csv(path, sep="\t", encoding='ISO-8859-1') if "S10" in path else pd.read_csv(path, sep="\t") for path in self.paths]
        self.data = pd.concat(dataframes, ignore_index=True)
        return self.data
    
    def prepare_dataset(self) -> DatasetDict:
        """_summary_

        Returns:
            DatasetDict: _description_
        """     
        copy_data = self.lire_data().copy()
        train_df = copy_data.sample(frac = 1, random_state = 42)
        train_dataset = Dataset.from_pandas(train_df)
        dataset_dict = DatasetDict({
            "train": train_dataset,
        })
        self.dataset_dict = dataset_dict.remove_columns(['__index_level_0__', 'ArticleTitle', 'DifficultyFromQuestioner', 'DifficultyFromAnswerer', 'ArticleFile'])

class EmbeddingsManager:
    """Une classe manager pour avoir les embeddings des textes"""
    def __init__(self, model_name = "bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.question_embeddings = None

    def get_embeddings(self, text):
        """_summary_

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Tokenisation avec retour en tenseurs
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Prendre la moyenne des embeddings sur la séquence
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def load_embedding(self, path = "question_embeddings.npy"):
        """_summary_

        Args:
            path (str, optional): _description_. Defaults to "question_embeddings.npy".
        """
        self.question_embeddings = np.load(path)

    def save_embeddings(self, embeddings, path = "question_embeddings.npy"):
        """_summary_

        Args:
            embeddings (_type_): _description_
            path (str, optional): _description_. Defaults to "question_embeddings.npy".
        """
        np.save(path, embeddings)
        self.question_embeddings = embeddings

class QuestionAnswerSystem:
    """Le systeme pour avoir la réponse à des questions en basant sur le cos-similarity"""
    def __init__(self, embedding_manager, questions, answers):
        self.embedding_manager = embedding_manager
        self.questions = questions
        self.answers = answers

    def encode_question(self):
        """_summary_"""        
        self.embedding_manager.question_embeddings = np.array(
            [self.embedding_manager.get_embeddings(q) for q in self.questions]
        )
        self.embedding_manager.save_embeddings(self.embedding_manager.question_embeddings)
    
    def find_answer(self, new_question):
        """_summary_

        Args:
            new_question (_type_): _description_

        Returns:
            _type_: _description_
        """
        new_question_embedding = self.embedding_manager.get_embeddings(new_question).reshape(1, -1)
        similarities = cosine_similarity(new_question_embedding, self.embedding_manager.question_embeddings).flatten()
        closest_question_idx = np.argmax(similarities)
        return self.answers[closest_question_idx]
    
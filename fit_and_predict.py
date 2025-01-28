import torch
from sentence_transformers import SentenceTransformer
import os

import spacy

import string

from sklearn.neighbors import NearestNeighbors

nlp = spacy.load("ru_core_news_sm")

stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation


def text_preprocess(sentence: str) -> str:
    """
    :param sentence: текст для предобработки
    :return: текст, состоящий из слов в начальной форме,
    без знаков пунктуации, прописными буквами, без слов, не несущих никакого смысла
    """
    doc = nlp(sentence)

    my_tokens: list = [word.lemma_.lower().strip() for word in doc]

    my_tokens: list = [word for word in my_tokens if word not in stop_words and word not in punctuations]

    sentence: str = " ".join(my_tokens)

    return sentence


class ClassificatorModel:
    def __init__(self, model_name: str, threshold: int | float = 0.5, neighbors_amount: int = 3):
        """
        Класс модели, который содержит методы: \n
        fit - обучение модели на основе существующих документов, разделенных на классы (по папкам!); \n
        predict - классификация новых документов: распределение их по существующим классам (или класс Unknown в случае,
        если не удалось определить принадлежность к конкретному классу) при помощи алгоритма ближайших соседей; \n
        score - вычисление доли точных (верных) предсказаний модели.
        :param model_name: Ссылка на модель HuggingFace
        :param threshold: Порог, после которого новый документ не будет считаться схожим с существующими
        :param neighbors_amount: Количество соседних документов для сравнения.
        """
        self.model = SentenceTransformer(model_name,
                                         similarity_fn_name="cosine",
                                         prompts={
                                             "clustering": "Определи тематику и основную мысль текста."})
        self.embeddings: list = []
        self.labels: list = []
        self.threshold: int | float = threshold
        self.knn = NearestNeighbors(n_neighbors=neighbors_amount, metric='cosine')

    def fit(self, path_to_dir: str) -> None:
        """
        Метод для обучения модели. Считывает директорию с папками (т.е. классами) и текстовыми документами в них.
        Предобрабатывает их и считает эмбеддинги.
        :param path_to_dir: Путь до директории с классами.
        :return: None.
        """
        for label in os.listdir(path_to_dir):
            label_dir: str = os.path.join(path_to_dir, label)
            if os.path.isdir(label_dir):
                for filename in os.listdir(label_dir):
                    filepath: str = os.path.join(label_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = text_preprocess(f.read())
                            embedding = self.model.encode(text)
                            self.embeddings.append(embedding)
                            self.labels.append(label)
                    except UnicodeDecodeError as e:
                        print(f"Ошибка при чтении файла {filepath}: {e}")
                        continue

        self.knn.fit(self.embeddings)

        if self.knn.n_samples_fit_ < self.knn.n_neighbors:
            self.knn.n_neighbors = self.knn.n_samples_fit_

        return None

    def predict(self, path_to_dir: str, show_neighbours: bool = False) -> list[str]:
        """
        Метод для предсказания меток классов новых документов. Получает на вход директорию с текстами.
        Предобрабатывает их и считает эмбеддинги. Далее проверка принадлежности к существующим классам происходит при
        помощи kNN.
        :param show_neighbours: Если True - выведет в консоль название файла и количество соседей, разделенных по классам
        и удовлетворяющих условиям.
        :param path_to_dir: Путь до папки с текстами.
        :return: Предсказанные метки классов в виде ["Метка класса1", "Метка класса2", ...]
        """
        new_embeddings: list[torch.Tensor] = []
        file_dir: list[str] = os.listdir(path_to_dir)
        for file in file_dir:
            with open(f"{path_to_dir}/{file}", 'r', encoding='utf-8') as f:
                text = text_preprocess(f.read())
                new_embeddings.append(self.model.encode(text))

        predicted_labels: list[str] = []

        for i, new_embedding in enumerate(new_embeddings):
            classes_counter: dict[str, int] = {}
            distances, indices = self.knn.kneighbors([new_embedding])

            for j, distance in enumerate(distances[0]):
                if distance <= self.threshold:
                    current_label = self.labels[indices[0][j]]
                    classes_counter[current_label] = classes_counter.get(current_label, 0) + 1

            if show_neighbours:
                print(f"Файл: {file_dir[i]}, количество соседей по классам: {classes_counter}")

            if classes_counter:
                predicted_label: str = max(classes_counter, key=classes_counter.get)
                predicted_labels.append(predicted_label)
            else:
                predicted_labels.append("Unknown")

        return predicted_labels

    def score(self, predicted: list[str], true: list[str]) -> None | str:
        """
        Метод для вычисления доли верно предсказанных меток классов. Важно, чтобы в массиве true метки классов были
        расставлены в том порядке, в каком идут документы в директории. Иначе, точность будет высчитана неверно.
        :param predicted: Предсказанные метки.
        :param true: Ground truth метки.
        :return: Доля верно предсказанных классов.
        """
        pred_length: int = len(predicted)
        true_length: int = len(true)

        counter: int = 0

        if pred_length != true_length:
            raise ValueError(
                f"Длина предсказанных меток ({pred_length}) не соответствует длине правильных меток ({true_length}).")

        else:
            for i, v in enumerate(predicted):
                if v == true[i]:
                    counter += 1

            return f'Доля верно предсказанных классов: {round(counter / pred_length * 100, 2)}%'


model = ClassificatorModel("paraphrase-multilingual-MiniLM-L12-v2", 0.2, 5)

model.fit("train")

predicted = model.predict("test", True)

print(predicted)

true = ['Россия', 'Россия', 'Мир', 'Россия', 'Unknown', 'Unknown']

print(model.score(predicted, true))

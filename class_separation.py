from sentence_transformers import SentenceTransformer
import os

import torch
import spacy

import string

nlp = spacy.load("ru_core_news_sm")

stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation


def spacy_tokenizer(sentence):

    doc = nlp(sentence)

    my_tokens = [word.lemma_.lower().strip() for word in doc]

    my_tokens = [word for word in my_tokens if word not in stop_words and word not in punctuations]

    sentence = " ".join(my_tokens)

    return sentence


class ClassificatorModel:
    def __init__(self, model_name: str, threshold: int | float = 0.5):
        self.model = SentenceTransformer(model_name,
                                         similarity_fn_name="cosine",
                                         prompts={"clustering": "Определи тематику текста и на ее основе реши, относится ли текст к существующим классам или нет."})
        self.classes: dict = {}
        self.grouped_files: dict = {}
        self.threshold: torch.Tensor = torch.Tensor([threshold])
        self.dictionary_length: int = -1

    def predict(self, path_to_dir: str):
        for file in os.listdir(path_to_dir):
            with open(f"{path_to_dir}/{file}", encoding='utf-8') as f:
                text = f.read()
                text = spacy_tokenizer(nlp(text))
                embedding = self.model.encode(text)

                found_similar = False

                if self.classes:
                    for class_id, (known_embedding, _) in self.classes.items():
                        similarity = self.model.similarity(embedding, known_embedding)[0]
                        if similarity > self.threshold:
                            self.grouped_files[class_id].append(file)
                            found_similar = True
                            break

                if not found_similar:
                    self.dictionary_length += 1
                    self.classes[self.dictionary_length] = (embedding, self.dictionary_length)
                    self.grouped_files[self.dictionary_length] = [file]
        return self.grouped_files


model = ClassificatorModel("cointegrated/rubert-tiny2", 0.75)

print(model.predict('texts'))

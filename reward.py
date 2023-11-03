import os
import math
import json
import torch
import argparse
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from typing import List, Tuple, Dict
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from utils.utils import batchify, product_rewards

nlp = spacy.load("en_core_web_sm")


class Reward:
    def __init__(self, save_path: str, batch_size: int, device: int, params: argparse.Namespace):
        self.path = save_path
        self.batch_size = batch_size
        self.params = params
        self.device = f'cuda:{device}'

        cola_model_name = "textattack/roberta-base-CoLA"
        self.cola_tokenizer = RobertaTokenizer.from_pretrained(cola_model_name)
        self.cola_model = RobertaForSequenceClassification.from_pretrained(cola_model_name).to(self.device)

    def get_reward(self, prompts: List[str], responses: List[str], concepts: List[str], epoch: str) -> Dict[str, List[float]]:
        reward_dict = {'coverage': [], 'cola': []}

        for response, concept in tqdm(zip(responses, concepts), total=len(concepts), desc='computing coverage'):
            reward_dict['coverage'].append(self._compute_coverage(response, concept, use_binary=self.params.binary_coverage))

        if not self.params.binary_coverage:
            reward_dict['binary_coverage'] = [int(c == 1) for c in reward_dict['coverage']]

        for texts in tqdm(batchify(responses, self.batch_size), total=math.ceil(len(responses) // self.batch_size),
                          desc='scoring generations'):

            texts = [t.strip() for t in texts]
            inputs = self.cola_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.cola_model(**inputs).logits
                probs = logits.softmax(dim=-1)
            scores = probs[:, 1].tolist()
            reward_dict['cola'].extend(scores)

        overall_reward = product_rewards([reward_dict['coverage'], reward_dict['cola']])
        reward_dict.update({'reward': overall_reward})

        zip_scores = list(zip(reward_dict['coverage'], reward_dict['cola']))
        data = pd.DataFrame.from_dict({'prompt': prompts, 'concepts': concepts})
        collate(data, responses, zip_scores, os.path.join(self.path, f'reward_{epoch}.json'))

        return reward_dict

    @staticmethod
    def _compute_coverage(output, concept, use_binary=False):
        lematized_concepts = [nlp(c.strip())[0].lemma_ for c in concept.split('-')]
        lemmatized_output = []
        for token in output.strip().split():
            lemmatized_output.extend([x.lemma_ for x in nlp(token)])

        if use_binary:
            score = 0
            for word in lematized_concepts:
                if word in lemmatized_output:
                    score += 1

            if score < len(lematized_concepts):
                return 0
            ordered_concept = sorted(lematized_concepts, key=lambda x: lemmatized_output.index(x))
            return int(ordered_concept == lematized_concepts)

        else:
            output_keywords = []
            for token in lemmatized_output:
                if token in lematized_concepts and token not in output_keywords:
                    output_keywords.append(token)
            assert len(output_keywords) <= len(lematized_concepts), f'concepts: {concept}, keywords: {output_keywords}'

            coverage = 0
            for i in range(len(output_keywords)):
                if lematized_concepts[i] == output_keywords[i]:
                    coverage += 1
                else:
                    break
            return coverage / len(lematized_concepts)


def collate(dataset: pd.DataFrame,
            generations: List[str],
            scores: List[Tuple],
            output_file: str = ''):
    generations_col_iter = [{'text': g, 'coverage': s[0], 'cola': s[1]} for g, s in zip(generations, scores)]

    assert len(generations) % len(dataset) == 0
    n = len(generations) // len(dataset)
    generations_col = list(batchify(generations_col_iter, n))
    dataset['generations'] = generations_col

    if output_file:
        dataset.to_json(output_file, orient='records', lines=True)

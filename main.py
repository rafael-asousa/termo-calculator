from collections import Counter
from typing import List
import numpy as np
from unidecode import unidecode

import pandas as pd
import tqdm


class WordleOptimizer:
    def __init__(self) -> None:
        self.guesses_path = '/home/rafael/Downloads/portfolio/Estudo/termo/word_list.txt'
        self.answer_path  = '/home/rafael/Downloads/portfolio/Estudo/termo/answer_list.txt'
    
    @property
    def possible_guesses(self) -> List:
        with open(self.guesses_path, 'r+') as f:
            guesses = f.read()

        guesses = self.clear_list(guesses)

        return [*guesses, *self.possible_answers]

    @property
    def possible_answers(self) -> List:
        with open(self.answer_path, 'r+') as f:
            answers = f.read()

        answers = self.clear_list(answers)

        return answers

    def clear_list(self, list_to_clear: List) -> List:

        final_list = list_to_clear.replace('\"', '').split(',')
        final_list = [unidecode(element) for element in final_list]
        
        return final_list

    def entropy(self, word: str, possible_answers) -> float:

        states = np.zeros(tuple((3,3,3,3,3)))

        for other_word in possible_answers:
            states[tuple(self.compare_words(word, other_word))] += 1

        states = states[states != 0]

        total_states = np.sum(states)
        prob_states = states/total_states

        return np.sum(prob_states * np.log2(prob_states))*(-1)

    def compare_words(self, word1: str, word2: str) -> List[int]:
        counts = Counter(word2)
        indexes = [0 for _ in range(len(word1))]
        for i, letter in enumerate(word1):
            if word2[i] == letter:
                indexes[i] = 2
                counts[letter] -= 1
        for i, letter in enumerate(word1):
            if counts[letter] > 0 and not indexes[i]:
                indexes[i] = 1
                counts[letter] -= 1
    
        return indexes

    def rank_words(self):
        word_ranking = {}
        possible_answers = self.possible_answers

        for word in tqdm.tqdm(self.possible_guesses):
            word_ranking[word] = self.entropy(word, possible_answers)

        return pd.DataFrame.from_dict(word_ranking, orient='index', columns=['entropy'])

if __name__ == '__main__':
    def main():
        wordle = WordleOptimizer()
        wordle.rank_words().to_csv('./ranking_1602.csv')

    main()
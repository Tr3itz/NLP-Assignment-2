import re
import string

from math import sqrt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer


# Remove punctuation from the string using regular expression
def remove_punctuation(text: str):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub("", text)


# Text normalization pipeline
def text_normalization(text: str) -> list[str]:
    stop_list: list[str] = stopwords.words('english')
    ps = PorterStemmer()

    text: str = remove_punctuation(text)

    # Tokenization -> removing tokens of length 1 such as single characters or other residues
    tokens: list[str] = [token.lower() for token in word_tokenize(text) if len(token) > 1]

    # Stop words removal
    tokens = [token for token in tokens if token not in stop_list]

    # Stemming
    stems: list[str] = [ps.stem(token) for token in tokens]

    return stems


# Create the bag of words of a text
def bow_creation(text: str) -> dict:
    bow: dict = dict()

    words: list[str] = text_normalization(text)

    for w in words:
        if w not in bow.keys():
            bow[w] = 1
        else:
            bow[w] += 1

    return bow


# Normalize the bow of a given slice of the document
def bow_normalization(pool: dict, _slice: dict) -> dict:
    length: int = sum(pool.values())

    norm_bow: dict = dict()

    for word in pool.keys():
        if word in _slice.keys():
            norm_bow[word] = _slice[word]/length
        else:
            norm_bow[word] = 0

    return norm_bow


def does_fit(sentence: str, _slice: list[str], window: int):
    return len(word_tokenize(' '.join(_slice))) + len(word_tokenize(sentence)) <= window


def squared_sum(vect: list[float]) -> float:
    return round(sqrt(sum([a*a for a in vect])), 3)


# Returns the cosine similarity between two slices
def cos_similarity(pool: dict, slice1: str, slice2: str) -> float:
    norm_slice1 = bow_normalization(pool, bow_creation(slice1))
    norm_slice2 = bow_normalization(pool, bow_creation(slice2))

    numerator = sum(a * b for a, b in zip(norm_slice1.values(), norm_slice2.values()))
    denominator = squared_sum(norm_slice1.values()) * squared_sum(norm_slice2.values())

    return round(numerator/float(denominator), 3)


class Document:
    def __init__(self, file: str):
        with open(file, 'r') as f:
            self.text: str = f.read()  # text of the file
            self.sentences: list[str] = sent_tokenize(self.text)  # list of sentences of the text
            self.length: int = len(word_tokenize(self.text))  # length of the text in terms of tokens
            self.pool: dict = bow_creation(self.text)  # bag of words of the text

    def slice_document(self, window: int) -> list[str]:

        #  if the text already fits the context window
        if self.length <= window:
            return [self.text]

        # slices of the text
        slices: list[str] = []

        # current analyzed slice -> "window" of sentences that scrolls down through the text
        current_slice: list[str] = []

        # slice both previous and adjacent to the current slice
        adjacent_slice: list[str] = []

        # slices are built sentence by sentence
        for s in self.sentences:
            if len(current_slice) == 0:
                current_slice.append(s)
                continue

            # if there exists an adjacent slice...
            if len(adjacent_slice) > 0:

                # ...and if the current slice is too similar to the previous one...
                if cos_similarity(self.pool, ' '.join(adjacent_slice), ' '.join(current_slice)) > 0.8:

                    # "scroll down" down the window:
                    # make room in the current slice in order to fit the new sentence
                    while not does_fit(s, current_slice, window):
                        current_slice.pop(0)  # remove sentences for the top

                    current_slice.append(s)
                    continue

            slice_token_size = len(word_tokenize(' '.join(current_slice)))
            sent_token_size = len(word_tokenize(s))

            # if adding the sentence to the current slice exceeds the context window...
            if slice_token_size + sent_token_size > window:
                slices.append(' '.join(current_slice))  # add the current slice to the list of slices
                adjacent_slice = current_slice.copy()  # set the previous slice as the last added one

                # "scroll down" down the window:
                # make room in the current slice in order to fit the new sentence
                while not does_fit(s, current_slice, window):
                    current_slice.pop(0)  # remove sentences for the top

                current_slice.append(s)
            else:
                current_slice.append(s)

        # if there is something left, make it different enough form the last added slice, then add it
        while cos_similarity(self.pool, slices[-1], ' '.join(current_slice)) > 0.8:
            current_slice.pop(0)

            # if the remaining slice has emptied, it would've been too similar to the last added slice
            if len(current_slice) == 0:
                break

        if len(current_slice) != 0:
            slices.append(' '.join(current_slice))

        return slices

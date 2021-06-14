from typing import Union, Sequence, List
from smart_open import open
import pandas as pd
from tqdm import tqdm

import gensim
from gensim.parsing.preprocessing import STOPWORDS

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def nltk_tag2wordnet(nltk_tag: str) -> Union[str, None]:
    """
    Converts nltk part of speech (POS) tag to wordnet
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


class YelpReviewsCorpus:
    """
    Memory-efficient corpus generator from the Yelp reviews dataset,
    filtered by specified business ids and rating (stars)
    """

    def __init__(self, filepath: str, business_id: Sequence[str], stars: int = None, custom_stopwords=None):
        """
        Args:
            filepath: path of the yelp reviews json file
            business_id: reviews only for specified businesses will be filtered
            stars: filter reviews that have specific rating
            custom_stopwords: additional specific stopwords to be removed from the documents
        """
        if custom_stopwords is None:
            custom_stopwords = {}
        self.filepath = filepath
        self.dictionary = gensim.corpora.Dictionary()
        self.business_id = business_id
        self.stars = stars
        self.counter = 0
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = STOPWORDS.union(custom_stopwords)

    def _lemmatize(self, tokens: Sequence[str]) -> List[str]:
        """
        Lemmatize sequence of tokens with POS (part of speech) analysis
        """
        pos_tagged = nltk.pos_tag(tokens)
        lemmatized = [(self.lemmatizer.lemmatize(x[0], nltk_tag2wordnet(x[1]))
                       if nltk_tag2wordnet(x[1]) is not None else x[0]) for x in pos_tagged]
        return lemmatized

    def _text2bow(self, text: str):
        """
        Preprocess text document (tokenize, remove stopwords, lemmatize),
        Calculates "bag of words" corpus, populates local dictionary
        """
        # lowercase, tokenize, filter short words
        tokens = gensim.utils.simple_preprocess(text)

        # remove stopwords
        tokens = [x for x in tokens if x not in self.stopwords]

        # lemmatize
        lemmatized_tokens = self._lemmatize(tokens)

        # generate bag of words
        bow = self.dictionary.doc2bow(lemmatized_tokens, allow_update=True)
        return bow

    def __iter__(self):
        with open(self.filepath) as f:
            reader = pd.read_json(f, orient='records', lines=True, chunksize=1000)
            for chunk in tqdm(reader, desc='Loading reviews'):
                reviews = chunk[chunk.business_id.isin(self.business_id)
                                & ((chunk.stars == self.stars) if self.stars is not None else True)]

                for text in reviews.text:
                    self.counter += 1
                    print(f" Processed gym's reviews: {self.counter}", end="\r")
                    yield self._text2bow(text)

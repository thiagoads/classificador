import re
from collections import Counter
from nltk.stem import RSLPStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize

class MyCustomTokenizer():
    
    def __init__(self, stopwords = [], stemmer = RSLPStemmer()):
        self._stopwords = stopwords
        self._stemmer = stemmer
        
    def preprocess(self, content):
        content = re.compile('<[^<>]+>').sub(' ', content)
        #TODO tratar emojis
        content = re.compile('[0-9]{1,2}\\/[0-9]{1,2}\\/[0-9]{4}').sub(' datemask ', content)
        content = re.compile('[0-9]+').sub(' numbermask ', content)
        content = re.compile('(http|https)://[^\s]*').sub(' httpaddrmask ', content)
        content = re.compile('[^\s]+@[^\s]+').sub(' emailaddrmask ', content)
        content = re.compile('[$]+').sub(' dollarmask ', content)
        return content
    
    def tokenize(self, data):
        tokenizer = RegexpTokenizer(r'\w+').tokenize
        corpus = []
        tokens = []
        for text in data:
            text = self.preprocess(text)
            filtered_tokens = []
            text_tokenized = tokenizer(text)
            for word in text_tokenized:
                if str.lower(word) not in self._stopwords:
                    token = str.lower(word)
                    token = self._stemmer.stem(token)
                    tokens.append(token)
                    filtered_tokens.append(token)
            text_processed = " ".join(filtered_tokens)
            corpus.append(text_processed)
        return corpus, tokens


"""
Extrai vocabulário de uma lista de tokens identificados na etapa de pré-processamento

Parametros:
-----------

tokens: list
    lista com tokens identificados
    
most_frequent: bool (optional)
    se informado (True), limita tamanho do vocabulário
    
top_number: int (optional)
    se informado juntamente com most_frenquent = True, 
    limita tamanho do vocabulário ao número informado
    
Retorno:
-----------
    vocabulary: list
        lista com o vocabulário final
        
"""
def extract_vocabulary_from_tokens(tokens, most_frequent: bool = False, top_number: int = 0):
    counter = Counter(tokens)
    if most_frequent == True:
        assert top_number > 0, "top_number deve ser maior que zero quanto most_frequent é True"
        sorted_tokens = sorted(counter.items(), key = lambda x: x[1], reverse = True)
        top_number_sorted_tokens = []
        limit = min(top_number, len(sorted_tokens))
        for i in range(limit):
            top_number_sorted_tokens.append(sorted_tokens[i][0])
        vocabulary = set(top_number_sorted_tokens)
    else:
        vocabulary = set(tokens)
    return vocabulary

import numpy as np
import warnings

"""
Calcula dimensão com base em lista de quantidade de tokens por observação de modo a resultar 
na menor perda de informação e sem tornar o vetor de representação final demasiadamente grande.

Parametros:
-----------

corpus_tokenized_size_list: list
    lista com quantidade de tokens para cada observação do conjunto

percent: int (optional | default = 95)
    usado para calcular o valor que representa o percentil informado dos dados

    
Retorno:
-----------
    dimension: int
        dimensão calculada para o tamanho de um vetor de representação vetorial
"""
def calculate_dimension_of_sentences(corpus_tokenized_size_list: list, percent: int = 95):
    mean = int(corpus_tokenized_size_list.mean())
    median = int(np.median(corpus_tokenized_size_list))
    percentile = int(np.percentile(corpus_tokenized_size_list, percent))
    average = int((mean + median) / 2)
    return int(max(average, percentile))

"""
Converte um conjunto de tokens para a representação vetorial de embeddings no modelo informado .

Parametros:
-----------

corpus_tokenized: list
    lista de tokens a ser convertida para embeddings

embeddings_model: gensim model
    modelo de embeddings treinado em um vocabulário

    
Retorno:
-----------
    corpus_embeddings: list
        lista de embeddings da lista de tokens informada
"""
def convert_tokens_to_embeddings(corpus_tokenized : list, embeddings_model):
    corpus_embeddings = []
    empty_documents = []
    tokens_not_found = {}
    assert len(corpus_embeddings) == 0, "lista a ser preenchida corpus_embeddings deve estar vazia!"
    for index, document_tokenized in enumerate(corpus_tokenized):
        if len(document_tokenized) == 0:
            empty_documents.append(index)
            vector_embeddings = np.zeros((0, embeddings_model.vector_size))
        else:
            try:
                vector_embeddings = embeddings_model.wv[document_tokenized]
            except KeyError:
                vector = []
                for token in document_tokenized:
                    if token in embeddings_model.wv:
                        vector.append(embeddings_model.wv[token])
                    else:
                        tokens_not_found.setdefault(index,[]).append(token)
                vector_embeddings = np.zeros((len(vector), embeddings_model.vector_size))
                if len(vector) > 0:
                    vector_embeddings[:len(vector), :] = vector[:]
                else:
                    vector_embeddings = np.zeros((0, embeddings_model.vector_size))
        corpus_embeddings.append(vector_embeddings)
    my_warnings = []
    if len(empty_documents) > 0:
        my_warnings.append("Total de {} documento(s) não possuem token algum: {}.".format(len(empty_documents), empty_documents))
    if len(tokens_not_found) > 0:
        my_warnings.append("Total de {} documento(s) com tokens não encontrados: {}.".format(len(tokens_not_found), tokens_not_found))
    if len(my_warnings) > 0:
        for w in my_warnings:
            warnings.warn(w)
    return corpus_embeddings

"""
Padrozina vetores de embeddings de diferentes tamanhos para ter apenas um vetor com o valor médio dos tokens.

Parametros:
-----------

data: list
    lista de embeddings a ser padronizada pela média

vector_size: int
    tamanho do vetor de representação usado no modelo de embeddings

    
Retorno:
-----------
    data_avg: ndarray
        lista de embeddinds patronizada de spahe len(data) x vector_size
"""
def apply_average_in_embeddings(data: list, vector_size: int):
    data_avg = []
    for vector in data:
        if vector.size:
            data_avg.append(vector.mean(axis=0))
        else:
            data_avg.append(np.zeros(vector_size, dtype=float))
    return np.array(data_avg, dtype=float)


"""
Padrozina vetores de embeddings de diferentes tamanhos para terem o mesmo tamanho completando dimensões faltante com padding.

Parametros:
-----------

data: list
    lista de embeddings a ser padronizada com uso de paddings

vector_size: int
    tamanho do vetor de representação usado no modelo de embeddings

dimension: int
    tamanho da dimensão do tensor final
    
Retorno:
-----------
    X: ndarray
        lista de embeddings padronizados com paddings de shape len(data) x dimension x vector_size
"""
def apply_padding_in_embeddings(data: list, vector_size: int, dimension: int):
    X = np.zeros((len(data), dimension, vector_size))
    for i, sequence in enumerate(data):
        limit = min(dimension, len(sequence))
        X[i, :limit, :] = sequence[:limit]
    return np.array(X, dtype=float)
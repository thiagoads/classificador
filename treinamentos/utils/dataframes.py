import numpy as np

"""
Cria novo dataframe com colunas de tokenização a partir de dataframe passado

Parametros:
-----------

original: pd.Dataframe
    dataframe original a partir do qual será criado um novo dataframe com colunas de tokenização

data_tokenized: list
    resultado da tokenização de um texto bruto, deve ter o mesmo número de linha do dataframe original
    
drop_size_zero: bool
    quando True, linhas do novo dataframe que não possuim tokens
    
Retorno:
-----------
    data: pd.Dataframe
        novo dataframe com colunas de tokenização
"""
def compute_tokenized_columns_in_dataframe(original, data_tokenized, drop_size_zero = True):
    data = original.copy(deep=True)
    data['Texto_Processado'] = data_tokenized
    data['Texto_Tokenizado'] = [s.split() for s in data_tokenized]
    data['Numero_de_Tokens'] = [len(s.split()) for s in data_tokenized]
    if drop_size_zero == True:
        data.drop(data[data['Numero_de_Tokens'] == 0].index, inplace = True)
    return data

"""
Cria novo dataframe com colunas de para análise de sentimento a partir de dataframe passado

Parametros:
-----------

original: pd.Dataframe
    dataframe original a partir do qual será criado um novo dataframe com colunas de código do assunto
    
Retorno:
-----------
    data: pd.Dataframe
        novo dataframe com colunas de sentimento
"""
def mapping_str_class_to_target(dataset, source: str = 'Assunto', target: str = 'Target'):
    data = dataset.copy(deep=True)
    data[target] = data[source].astype('category').cat.codes
    return data
def print_confusion_matrix_as_table(reports = None, model_names = None, show=False):
    assert reports != None, "Informe os resultados das avaliações!"
    assert model_names != None, "Informe os nomes dos modelos!"
    assert len(reports) == len(model_names), "reports e model_names devem ter mesmo tamanho."
    row_models = '|      '.ljust(21)
    header = '| Classe'.ljust(21)
    separator = '---'
    clazz_keys = [x for x in reports[0]]
    for model in model_names:
        row_models += ' | | *{}* | '.format(model)
        header += ' | Precision | Recall | F1-Score'
        separator += ' | --- | --- | ---'    
    row_models += '| '
    header += '| Support '
    separator += '| ---'

    output = header      + '\n' \
            + separator  + '\n' \
            + row_models + '\n' 

    if show:
        print(header)
        print(separator)
        print(row_models)

    for key in clazz_keys:
        row = '| {:.19s}  '.format(key.ljust(19))
        support = ' '
        for index, model_name in enumerate(model_names):
            dictionary = reports[index]
            dictionary2 = dictionary[key]
            if(type(dictionary2) is dict):
                precision = '{:.3f}     '.format(dictionary2['precision'])
                recall =  '{:.3f}  '.format(dictionary2['recall'])
                f1_score =  '{:.3f}    '.format(dictionary2['f1-score'])
                row = row + '| ' + precision
                row = row + '| ' + recall
                row = row + '|  ' + f1_score
                support = str(dictionary2['support'])
            else:
                accuracy = '{:.3f}    '.format(dictionary[key])
                row = row + '|           |        | ' + accuracy
        row = row + '| ' + support + ' |'

        output += row + '\n'
        if show:
            print(row)

    return output
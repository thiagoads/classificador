from flask import Flask
from flask import request, jsonify
from flask_restx import Api, Resource, fields

import tensorflow as tf
import numpy as np # definição de modelo que será validado ao receber post
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification 

app = Flask(__name__)
api = Api(app, version='1.0', title='Classificador API', description='Documentação da API do classificador')

# Define a namespace
ns = api.namespace('api', description='Operações')

# Define a data model for the request
request_model = api.model('Request', {
    'text': fields.String(required=True, description='Texto')
})

path_to_model = './model'

tokenizer = AutoTokenizer.from_pretrained(path_to_model)

model = TFAutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels = 10)

classes = ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4', 
           'Classe 5', 'Classe 6', 'Classe 7', 'Classe 8', 'Classe 9']


@ns.route('/predict')
class ApiResource(Resource):
    @api.doc(description='Classificar manifestação a partir de um texto')
    @api.expect(request_model)
    def post(self):
        try:
            data = request.json
            text = data.get('text')
            tokenized_text = tokenizer(text, 
                                        return_tensors = "np", 
                                        padding = True, 
                                        truncation=True, 
                                        max_length=512)
        
            logits = model.predict(dict(tokenized_text), verbose = False)['logits']
            probs = tf.nn.softmax(logits)
            index = np.argmax(probs, axis = 1)[0]
            predict = classes[index]
            softmax = probs.numpy()[0][index]
            return {'predict': predict, 'softmax': float(softmax)}, 200
        except Exception as e:
            return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)

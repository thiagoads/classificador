from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import os

model_huggingface_url = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(model_huggingface_url)

model = TFAutoModelForSequenceClassification.from_pretrained(model_huggingface_url, num_labels = 10)
model.summary()

path_to_model = './src/model'

if os.path.isdir(path_to_model) == False:
    os.mkdir(path_to_model)

tokenizer.save_pretrained(path_to_model)
model.save_pretrained(path_to_model)

loaded_tokenizer = AutoTokenizer.from_pretrained(path_to_model)
loaded_model = TFAutoModelForSequenceClassification.from_pretrained(path_to_model)

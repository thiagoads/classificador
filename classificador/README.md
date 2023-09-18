## criar venv do zero
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## ativar o ambiente criado
source .venv/bin/activate

## atualizar modelo local
python script.py

## rodar api em desenvolvimento
flask --app app --debug  run

## buildar imagem docker
./build.sh

## buildar imagem docker (manualmente)
docker build -t classificador .

## startar imagem docker
docker run -p 5000:5000 -d --name classificador classificador

## visualizar api pelo swagger
http://127.0.0.1:5000/

## classificar chamando api
curl -X POST -H 'Content-Type: application/json' -d @sample.json http://127.0.0.1:5000/api/predict
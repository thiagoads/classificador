#!/bin/bash

# ativar o ambiente
source .venv/bin/activate

# baixa o modelo do huggingface.co
python script.py

# constroi imagem do classificador
docker build -t classificador .
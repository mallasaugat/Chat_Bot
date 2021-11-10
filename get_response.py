from nltk_utils import preprocess_sentence, vectorize_new
from model import Net
import pickle
import json
import torch
import random


with open('intents.json','r',encoding='utf-8') as f:
    intents = json.load(f)

with open('vect.pkl','rb') as f:
    vect = pickle.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

n_inputs = data['n_inputs']
hidden1 = data['hidden1']
hidden2 = data['hidden2']
n_outputs = data['n_outputs']
model_state = data["model_state"]
tags = data["tags"]
model = Net([n_inputs,hidden1,hidden2,n_outputs])
model.load_state_dict(model_state)
model.eval()

def get_response(sentence):
    sentence = preprocess_sentence(sentence)
    X = vectorize_new(vect, sentence)
    X = X.reshape(1, -1)
    X = torch.from_numpy(X).float()
    z = model(X)
    _, yhat = torch.max(z, dim=1)
    probs = torch.softmax(z, dim=1)
    prob = torch.max(probs)
    tag = tags[yhat]
    if prob.item() < 0.75:
        return "Please Enter a correct sentence"
    else:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
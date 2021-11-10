
from get_response import get_response

bot_name = "Assistant"


while True:
    sentence = input("Vit Bot: ")
    if sentence == 'stop':
        break
    print(bot_name + ': ' + get_response(sentence))
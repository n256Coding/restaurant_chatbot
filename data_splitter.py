import json
import math

def divide_chunks(l, n): 

    size = len(l)
    chunk_size = math.ceil(size / n)

    for i in range(0, len(l), chunk_size):  
        yield l[i:i + chunk_size] 

with open('data/intents.json', 'r', encoding='utf-8') as g:
    original_file = json.load(g)
    original_intents = original_file.get('intents')
    output_json = []

    for intent in original_intents:
        tag = intent.get('tag')
        questions = intent.get('questions')
        answers = intent.get('answers')

        for i, element in enumerate(output_json):
            if element.get('tag', '') == tag:
                for question in questions:
                    if question not in element.get('questions'):
                        element.get('questions').append(question)
                output_json[i] = element
        else:
            output_json.append(intent)

    print(f'Number of tags: {len(output_json)}')
    print('Splitting..')

    splited_dataset = list(divide_chunks(output_json, 4))

    for i, set in enumerate(splited_dataset):
        with open(f'data/intents.split{i+1}.json', 'w', encoding='utf-8') as h:
            json.dump({
                'intents': set
            }, h)

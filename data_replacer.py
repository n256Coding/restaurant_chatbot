import json

with open('data/Chat bot_dataset.json', 'r', encoding='utf-8') as f:
    with open('data/intents.json', 'r', encoding='utf-8') as g:
        with open('data/intents.modified.json', 'w', encoding='utf-8') as h:
            edited_file = json.load(f)
            original_file = json.load(g)

            original_intents = original_file.get('intents')

            for element in edited_file.get('intents'):
                tag = element.get('tag')

                for element2 in original_intents:
                    if tag == element2['tag']:
                        element2['answers'] = element['answers']

            output_json = {
                'intents': original_intents
            }

            json.dump(output_json, h)

print('done!')
import json

merged_list = []

for i in range(1, 5):
    with open(f'data/intents.split{i}.json', encoding='utf-8') as g:
        json_file = json.load(g)
        intent_list = json_file.get('intents')
        merged_list.append(intent_list)

merged_json = {
    "intents": merged_list
}

with open('data/intents.merged.json', 'w', encoding='utf-8') as f:
    json.dump(merged_json, f)
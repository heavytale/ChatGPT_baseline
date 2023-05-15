from pathlib import Path
import json


wow_path = r"C:\onedrive\_\NLPlab\Project_ACL2023\wizard_of_wikipedia"
wow_file = "test_random_split_v2.json"

# make the list of (checked_passage, prefixed_dialog_history, label)
wow_examples = []
with open(Path(wow_path) / wow_file, 'rt') as f:
    wow_json = json.load(f)
for i, topic_dialog in enumerate(wow_json):
    prefixed_dialog_history = []
    for j, dialog in enumerate(topic_dialog['dialog']):
        if dialog['speaker'].endswith('Apprentice'):
            prefixed_dialog_history.append('Human: ' + dialog['text'])
        elif dialog['speaker'].endswith('Wizard'):
            checked_passage = list(dialog['checked_passage'].values())
            if len(checked_passage) == 0:
                checked_passage = list(dialog['checked_sentence'].values())  # checked_passage doesn't exists
            checked_passage = checked_passage[0]
            wow_examples.append((checked_passage, prefixed_dialog_history, dialog['text']))
            prefixed_dialog_history.append('Agent: ' + dialog['text'])
        else:
            raise ValueError('dataset error')
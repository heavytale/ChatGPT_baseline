import itertools
import json
import os
import pickle
from copy import copy
import asyncio
import time
import sys
from pathlib import Path
from datetime import datetime

import nltk
import openai
import evaluate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm


def parse_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        for line in file.readlines():
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])

            if int(dialog_idx) == 1:
                data.append({'persona_info': [], 'partner_info': [], 'dialog': []})

            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['persona_info'].append(persona_info)

            elif dialog_line[0].startswith("partner's persona:"):
                persona_info = dialog_line[0].replace("partner's persona: ", "")
                data[-1]['partner_info'].append(persona_info)

            elif len(dialog_line) > 1:
                data[-1]['dialog'].append(dialog_line[0])
                data[-1]['dialog'].append(dialog_line[1])

        return data


# make prompt for chatGPT
prompt_example = f"""
nightly modifying...
"""


def pchat_prompt_label_generator(p_keywords, dialogues, history_size=3):
    assert len(p_keywords) == len(dialogues)
    for p_keyword, dialogue in zip(p_keywords, dialogues):
        prefixed_dialogue = []
        for i, utterance in enumerate(dialogue):
            if i % 2 == 0:
                prefix = 'User: '
            else:
                prefix = 'Agent: '
            prefixed_dialogue.append(prefix + utterance)

        self_p_keywords = [[m for m in p['self'] if not m == 'i'] for p in p_keyword]
        self_p_keywords_str = '\n'.join([f'persona {i+1}: ' + ', '.join(self_p_keyword) for i, self_p_keyword in enumerate(self_p_keywords)])
        partner_p_keywords = [[m for m in p['partner'] if not m == 'i'] for p in p_keyword if not p['partner'] == 'i']
        partner_p_keywords_str = '\n'.join([f'persona {i+1}: ' + ', '.join(partner_p_keyword) for i, partner_p_keyword in enumerate(partner_p_keywords)])

        # chunk utterances num with 2, 4, 6 ...
        for i in range(0, len(prefixed_dialogue), 2):
            joined_dialogue = "\n".join(prefixed_dialogue[max(0, i+1-history_size):i+1])

            prompt = f"""You are an engaging and friendly conversational agent who has a consistent personality.
Being an agent, make a brief utterance using the persona keywords given.

Agent's persona keywords:

{self_p_keywords_str}

User's persona keywords:

{partner_p_keywords_str}

Dialogue:

{joined_dialogue}
Agent: """
            label = dialogue[i+1]
            yield prompt, label


def focus_prompt_label(p_keywords, landmark_name, dialog_history, label, history_size=3):
    prefixed_dialogue_history = []
    for i, sent in enumerate(dialog_history):
        if i % 2 == 0:
            prefixed_dialogue_history.append("User: " + sent)
        else:
            prefixed_dialogue_history.append("Agent: " + sent)

    # joined_p_keywords = ', '.join(p_keywords)
    joined_p_keywords = '\n'.join([f'persona {i+1}: ' +
                                   ', '.join(self_p_keyword) for i, self_p_keyword in enumerate(p_keywords)])
    prefixed_dialogue_history = '\n'.join(prefixed_dialogue_history[-history_size:])

    prompt = f"""You are a conversational agent, supporting users with knowledge-grounded and user-customized answer.
You must helpfully inform the user of the knowledge about the geographical landmark, considering the user's persona.
Being an agent, make a brief utterance using the persona keywords and knowledge given.

User's persona keywords:

{joined_p_keywords}

Landmark name: {landmark_name}

Dialogue:

{prefixed_dialogue_history}
Agent: """
    return prompt, label


def wow_prompt_label(checked_passage, prefixed_dialog_history, label, history_size=3):
    template = """You are an engaging and friendly conversation agent who uses knowledge as context.
Being an agent, make a brief utterance using the knowledge given.

Given knowledge: {checked_passage}

Dialogue:

{prefixed_dialog_history}
Agent: """
    prompt = PromptTemplate(
        input_variables=["checked_passage", "prefixed_dialog_history"],
        template=template
    )
    return prompt.format(
        checked_passage=checked_passage,
        prefixed_dialog_history='\n'.join(prefixed_dialog_history[-history_size:])
    ), label


def get_chatgpt_response(prompt):
    # Not using LangChain
    openai.api_key = "..."  # API Key
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stop="\n",
    )
    output = completion['choices'][0]['message']['content']
    usage = completion['usage']
    return output, usage


def request_with_save(prompts, save_dir):
    """ deprecated. use get_chatgpt_response instead to do asynchronously """
    # Not using LangChain
    # So it takes a time
    file = Path(save_dir) / f"chatgpt_responses{datetime.now().strftime('-%d-%H%M%S')}.pkl"
    results = []
    cumulative_tokens = 0
    iterator = tqdm(prompts)
    for prompt in iterator:
        output, usage = get_chatgpt_response(prompt)
        results.append((output, usage))
        cumulative_tokens += usage['total_tokens']
        with open(file, 'wb') as f:
            pickle.dump(results, f)
        iterator.set_description(f'cumulative_tokens={cumulative_tokens}')


def load_chatgpt_result(task, save_dir):
    if task not in ['pchat', 'focus', 'wow']:
        raise AssertionError
    chatgpt_pickles = list(Path(save_dir).glob(f"chatgpt-{task}-*.pkl"))
    datas = []
    for file in chatgpt_pickles:
        with open(file, 'rb') as f:
            datas.append(pickle.load(f))

    return list(itertools.chain(*datas))


def evaluate_results(chatgpt_output, labels):  # TODO Debug
    # preprocess
    chatgpt_output = ["\n".join(nltk.sent_tokenize(out)) for out in chatgpt_output]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    rouge = evaluate.load('rouge')
    chrf = evaluate.load('chrf')
    bleu = evaluate.load('bleu')
    sacrebleu = evaluate.load('sacrebleu')
    print("rouge")
    print(rouge.compute(predictions=chatgpt_output, references=labels))
    print("bleu")
    print(bleu.compute(predictions=chatgpt_output, references=labels))
    print("chrf")
    print(chrf.compute(predictions=chatgpt_output, references=labels, lowercase=True, word_order=2))
    print("sacrebleu")
    print(sacrebleu.compute(predictions=chatgpt_output, references=labels))


def persona_sent_to_persona_keywords(p_sent):
    p_keys = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(p_sent)) if
              pos[0] == 'N' or pos[0] == 'V']
    return p_keys


async def async_generate(llm, prompt):
    resp = await llm.agenerate([[HumanMessage(content=prompt)]])
    return resp


async def generate_concurrently(prompts):
    # llm = ChatOpenAI(max_tokens=50)  # TODO debug
    llm = ChatOpenAI(max_tokens=70, stop='\n')
    tasks = [async_generate(llm, prompt) for prompt in prompts]
    return await tqdm_asyncio.gather(*tasks)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_chatgpt_output_async(task, prompts, save_dir=".", time_sleep=20):
    if not os.environ['OPENAI_API_KEY']:
        print("os.environ['OPENAI_API_KEY'] not exists!")
        sys.exit(-1)
    iters = list(chunks(prompts, 100))
    for i, chunk in enumerate(iters):
        print(f"{i}th iterate")
        results = asyncio.run(generate_concurrently(chunk))
        file = Path(save_dir) / f"chatgpt-{task}-{i}.pkl"
        with open(file, 'wb') as f:
            pickle.dump(results, f)
        print(f"{i}th iteration save complete")
        if i + 1 != len(iters):
            time.sleep(time_sleep)  # not sleeps at last iteration


def do_chatgpt_async(task, prompts, save_dir):
    """save chatgpt results to save_dir"""
    save_dir.mkdir(exist_ok=True)
    if save_dir.is_dir() and not list(save_dir.glob('*')):
        get_chatgpt_output_async(task, prompts, save_dir)
    else:
        raise FileExistsError(f'{save_dir} is not empty')


def evaluate_by_chatgpt(use_api, task, prompts, labels):
    if task not in ['pchat', 'focus', 'wow']:
        print(f"WARNING: specified task is {task} which is not considered yet.")
    save_dir = Path(f'{task}_chatgpt_outputs')
    if use_api:
        do_chatgpt_async(task, prompts, save_dir)

    chatgpt_output = load_chatgpt_result(task, save_dir)
    chatgpt_str = [e.generations[0][0].text for e in chatgpt_output]
    evaluate_results(chatgpt_str, labels)


def do_pchat(use_api=False):
    # for p-chat dataset
    pchat_path = r"C:\OneDrive\_\NLPlab\Project_ACL2023\p-chat"
    test_persona = parse_data(os.path.join(pchat_path, "test_both_revised.txt"))

    pchat_train_raw_dpr = []

    for i, data in tqdm(enumerate(test_persona)):
        persona = data['persona_info']
        partner_persona = data['partner_info']
        combined_persona = list(zip(persona, partner_persona))

        for j, (p_sent, partner_p_sent) in enumerate(combined_persona):
            dialogid = 'pchat' + "_train_" + str(i) + "_" + "persona" + str(j)
            p_keys = persona_sent_to_persona_keywords(p_sent)
            partner_p_keys = persona_sent_to_persona_keywords(partner_p_sent)

            result = {'dataset': dialogid, 'p_keys': p_keys, 'partner_p_keys': partner_p_keys}

            pchat_train_raw_dpr.append(result)

    # make (dialogue number, persona keywords) list
    p_keywords_list = []
    dialogue_num = -1
    for example in pchat_train_raw_dpr:
        new_dialogue_num = int(example['dataset'].split('_')[2])
        if new_dialogue_num != dialogue_num:
            p_keywords_list.append([])
            dialogue_num = new_dialogue_num
        p_keywords_list[-1].append({
            'self': example['p_keys'],
            'partner': example['partner_p_keys']
        })

    dialogues = [p['dialog'] for p in test_persona]

    pchat_prompts_and_labels = list(pchat_prompt_label_generator(p_keywords_list, dialogues))
    pchat_prompts, pchat_labels = tuple(zip(*pchat_prompts_and_labels))

    # debug
    pchat_prompts = pchat_prompts  # TODO Debug
    pchat_labels = pchat_labels
    evaluate_by_chatgpt(
        use_api=use_api,
        task='pchat',
        prompts=pchat_prompts,
        labels=pchat_labels)


def do_focus(use_api=False):
    # for FoCus dataset
    focus_path = r"C:\OneDrive\_\NLPlab\Project_ACL2023\FoCus"
    focus_file = "valid_focus.json"

    # make the list of (persona, landmark_name, dialog_history, label)
    examples = []
    with open(Path(focus_path) / focus_file, 'rt') as f:
        focus_json = json.load(f)
    for dialog in focus_json['data']:
        persona_list = dialog['persona']
        landmark_name = dialog['landmark_link'].split('/wiki/')[1].replace('_', ' ')
        p_keywords = [persona_sent_to_persona_keywords(p_sent) for p_sent in persona_list]
        p_keywords = [[w for w in p if not w == 'i'] for p in p_keywords]

        # p_keywords = list(itertools.chain(*p_keywords))

        for utterance in dialog['utterance']:
            dialog_key = [key for key in utterance.keys() if 'dialogue' in key][0]
            dialogue_history = utterance[dialog_key]
            label = dialogue_history.pop()
            examples.append((p_keywords, landmark_name, dialogue_history, label))
    focus_examples = [focus_prompt_label(e[0], e[1], e[2], e[3]) for e in examples]

    focus_prompts, focus_labels = tuple(zip(*focus_examples))

    evaluate_by_chatgpt(
        use_api=use_api,
        task='focus',
        prompts=focus_prompts,
        labels=focus_labels)


def do_wow(use_api=False):
    # for Wizard of Wikipedia dataset
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
                wow_examples.append((checked_passage, copy(prefixed_dialog_history), dialog['text']))
                prefixed_dialog_history.append('Agent: ' + dialog['text'])
            else:
                raise ValueError('dataset error')

    wow_inputs = [wow_prompt_label(e[0], e[1], e[2]) for e in wow_examples]  # [(prompt, label), (prompt, label), ...]
    wow_prompts, wow_labels = tuple(zip(*wow_inputs))

    evaluate_by_chatgpt(
        use_api=use_api,
        task='wow',
        prompts=wow_prompts,
        labels=wow_labels)


def main():
    do_pchat(use_api=False)

    do_focus(use_api=False)

    do_wow(use_api=False)


if __name__ == "__main__":
    main()

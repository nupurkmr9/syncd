import argparse
from collections import defaultdict

import pandas as pd
import regex as re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPLATE_MESSAGE_SYSTEM = "You are a language model expert in suggesting image captions for different object categories."
TEMPLATE_MESSAGE_USER_BG_OBJECTS = '''Suggest {NUM_PROMPTS} caption for images of a {cat_}. The caption should provide a description of the background. DO NOT add any unnecessary adjectives or emotion words in the caption. Please keep the caption factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation. Follow these guidance for the captions: 
                                1. Generate captions of {cat_} in different backgrounds and scenes. 
                                2. Generate captions of {cat_} with another object in the scene.

                                Example captions for "White plastic bottle" are:
                                1. A white plastic bottle on a roadside cobblestone with stone bricks.
                                2. A white plastic bottle is placed next to a steaming cup of coffee on a polished wooden table. 

                                Example captions for "a blue truck" are:
                                1. A blue tank in a military storage facility with metal walls.
                                2. A blue tank on a desert battlefield ground, with palm trees in the background.
                                '''

TEMPLATE_MESSAGE_USER_BG_ANIMALS = '''Suggest {NUM_PROMPTS} caption for images of a {cat_}. Make sure "{cat_}" word is in the caption. The caption should provide a brief description of the background. DO NOT add any unnecessary adjectives or emotion words in the caption. Please keep the caption factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation. Follow these guidance for the captions: 
                            1. generate captions of {cat_} in different backgrounds and scenes.
                            2. generate captions of {cat_} with another object in the scene.
                            3. generate captions of {cat_} with different stylistic representations.

                            Example captions for the category "cat" are:
                            1. A photo of a siamese cat playing in a garden. The garden is filled with wildflowers. 
                            2. A cat is sitting beside a book in a library.
                            4. Painting of a cat in watercolor style. '''

TEMPLATE_MESSAGE_USER_ANIMALS = '''Suggest {NUM_PROMPTS} caption for images of a {cat_}. Make sure "{cat_}" word is in the caption. The caption should provide detailed visual information of the {cat_} including color and subspecies. DO NOT add any unnecessary adjectives or emotion words in the caption or background scene details. Please keep the caption factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation. 

                            Example captions for the category "cat" are:
                            1. The siamese cat has blue almond-shaped eyes and cream-colored fur with dark chocolate points on the ears, face, paws, and tail. 
                            2. The white fluffy Maine Coon cat with long, and bushy tail spread out beside it, and its thick fur has a mix of brown, black, and white stripes. 
                            3. The bengal cat with marbled coat features a pattern of vivid orange and black spots. '''


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")

terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]


def get_output(messages):
    texts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    inputs = tokenizer(texts, padding="longest", return_tensors="pt", padding_side='left')
    inputs = {key: val.cuda() for key, val in inputs.items()}
    temp_texts=tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

    gen_tokens = model.generate(
        **inputs, 
        max_new_tokens=2048, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    gen_text = [i[len(temp_texts[idx]):] for idx, i in enumerate(gen_text)]
    return gen_text


def clean_prompt(prompts):
    prompts = prompts.split('\n')
    prompts = [
        re.sub(r"[0-9]+", lambda num: "" * len(num.group(0)), prompt)
        for prompt in prompts
    ]
    prompts = [
        re.sub(r"^\.+", lambda dots: "" * len(dots.group(0)), prompt)
        for prompt in prompts
    ]
    prompts = [x.lower().strip().replace('"', "") for x in prompts]
    prompts = [x for x in prompts if x != '' and 'suggestions' not in x and 'note:' not in x and 'captions' not in x and '**' not in x and 'brief description' not in x and 'detailed visual' not in x and 'additional captions' not in x and 'meet your requirements!' not in x]
    
    prompts = list(set(prompts))
    return prompts


def get_prompts_deformable(categories, NUM_PROMPTS, batch_size):
    prompts = defaultdict(list)
    prompts_desc = defaultdict(list)
    with tqdm(total = len(categories)) as pbar:
        for i, cat_ in enumerate(categories):
            messages = [[
                    {"role": "system", "content": TEMPLATE_MESSAGE_SYSTEM},
                    {"role": "user", "content": TEMPLATE_MESSAGE_USER_BG_ANIMALS.format(NUM_PROMPTS=NUM_PROMPTS, cat_=cat_)},
                ] for j in range(batch_size)]
            gen_text = get_output(messages)
            output = [clean_prompt(x) for x in gen_text]
            prompts[cat_] = [x for y in output for x in y]
            messages = [[
                    {"role": "system", "content": TEMPLATE_MESSAGE_SYSTEM},
                    {"role": "user", "content": TEMPLATE_MESSAGE_USER_ANIMALS.format(NUM_PROMPTS=NUM_PROMPTS, cat_=cat_)},
                ] for j in range(batch_size)]
            gen_text = get_output(messages)
            output = [clean_prompt(x) for x in gen_text]
            prompts_desc[cat_] = [x for y in output for x in y]
            pbar.update(1)
    return prompts, prompts_desc


def get_prompts_rigid(categories, captions, NUM_PROMPTS, batch_size):
    prompts = defaultdict(list)
    cats_ = []
    class_prompts = []
    with tqdm(total = len(categories)) as pbar:
        for row in captions.iterrows():
            cat_, class_prompt = row[1].values
            cats_.append(cat_)
            class_prompts.append(class_prompt)
            if len(cats_) == batch_size:
                messages = [[
                        {"role": "system", "content": TEMPLATE_MESSAGE_SYSTEM},
                        {"role": "user", "content": TEMPLATE_MESSAGE_USER_BG_OBJECTS.format(NUM_PROMPTS=NUM_PROMPTS, cat_=class_prompts[j])},
                    ] for j in range(batch_size)]
                gen_text = get_output(messages)
                output = [clean_prompt(x) for x in gen_text]
                for j in range(batch_size):
                    prompts[cats_[j]] = output[j]
                cats_ = []
                class_prompts = []
                pbar.update(batch_size)
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(description="get prompts from LLM")
    parser.add_argument("--outdir", type=str, default="assets/generated_prompts")
    parser.add_argument("--captions", type=str, help='path to CAP3D csv. Required for objaverse rigid category captions')
    parser.add_argument('--rigid', action="store_true")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    if not args.rigid:
        with open('assets/categories.txt', 'r') as f:
            categories = f.readlines()
        categories = [x.strip() for x in categories]
        NUM_PROMPTS = 25
        batch_size = 6
        prompts, prompts_desc = get_prompts_deformable(categories, NUM_PROMPTS, batch_size)
        torch.save(prompts, f'{args.outdir}/prompts_deformable.pt' )
        torch.save(prompts_desc, f'{args.outdir}/prompts_desc_deformable.pt' )
    else:
        categories = list(torch.load('assets/objaverse_ids.pt'))
        captions = pd.read_csv(f'{args.captions}', header=None)
        mask = captions[0].isin(categories)
        captions = captions[mask]
        NUM_PROMPTS = 10
        batch_size = 10
        prompts = get_prompts_rigid(categories, captions, NUM_PROMPTS, batch_size)
        torch.save(prompts, f'{args.outdir}/prompts_objaverse.pt' )

if __name__ == "__main__":
    args = parse_args()
    main(args)

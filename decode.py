import torch
import json
from tqdm import tqdm
import numpy as np
from arguments import get_args
from policy_gp3 import Policy
from reward import Reward
from utils.utils import batchify
from utils.constants import HOME_PATH

batch_size = 12
datasets = json.load(open(f'{HOME_PATH}/data/commongen/test.json', 'r'))
args = get_args()
evaluator = Reward('output', 32, 0, args)

checkpoint_path = f'PATH_TO_ADAPTER_CHECKPOINT'
tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(5)]
checkpoint = torch.load(checkpoint_path, map_location='cpu')['value_model']
policy = Policy('gpt2-large', checkpoint, 'cuda', tree_tokens, 1.0, True)
print('initialization done')

prompts, concepts, constraints = [], [], []
for key, item in datasets.items():
    for order_words in item['human_order'][:1]:
        constraint = json.dumps([list(map(lambda x: policy.tokenizer.encode(f' {x}'), item['inflection'][w]))
                                 for w in order_words.split('-')])
        prompt = 'Generate a sentence including the following keywords in the same order as listed: %s\n\nAnswer:'
        prompt = prompt % ' '.join(order_words.split('-'))
        prompts.append(prompt)
        concepts.append(order_words)
        constraints.append(constraint)

generations = []
for batch in tqdm(batchify(zip(prompts, constraints), batch_size=batch_size), total=len(prompts) // batch_size):
    batch_prompts, batch_constraints = [b[0] for b in batch], [b[1] for b in batch]
    outputs = policy.sample(prompts=batch_prompts, constraints=batch_constraints,
                            max_len=64, top_p=0.9, use_control_code=True)
    generations.extend(outputs['response/text'])

score_dict = evaluator.get_reward(prompts, generations, concepts, 'eval_test')
for name, scores in score_dict.items():
    metric_score = np.mean(scores)
    print(f"  {name} = {metric_score:+.3f}")


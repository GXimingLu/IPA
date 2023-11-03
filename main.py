import os
import torch
import json
import time
import logging
import random
import argparse
import numpy as np
from typing import List
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from arguments import get_args
from policy import Policy
from data_pool import DataPool
from reward import Reward
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum
from utils.generation_utils import decode

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class PromptDataset(Dataset):
    def __init__(self, path, tokenizer):
        data = json.load(open(path, 'r'))
        self.items = [v for k, v in data.items() if v['human_order']]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        order_words = random.choice(item['human_order'])
        constraint = json.dumps([list(map(lambda x: self.tokenizer.encode(f' {x}'), item['inflection'][w]))
                                 for w in order_words.split('-')])
        prompt = 'Generate a sentence including the following keywords in the same order as listed: %s\n\nAnswer:'
        prompt = prompt % ' '.join(order_words.split('-'))

        return {
            'order': order_words,
            'constraint': constraint,
            'prompt': prompt,
        }


class PromptCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        concepts = [sequence['order'] for sequence in sequences]
        prompts = [sequence['prompt'] for sequence in sequences]
        constraints = [sequence['constraint'] for sequence in sequences]

        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return input_ids, attention_mask, concepts, constraints


class SequenceDataset(Dataset):
    def __init__(self, data_pool: DataPool):
        self.queries, self.responses, self.cat_tokens = data_pool.get_data()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {'query': self.queries[idx],
                'response': self.responses[idx],
                'cat_tokens': self.cat_tokens[idx]
                }


class SequenceCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        queries = [sequence['query'] for sequence in sequences]
        responses = [sequence['response'] + self.tokenizer.eos_token for sequence in sequences]
        cat_ids = [self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens']) for sequence in sequences]

        query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = query_encodings_dict['input_ids']
        query_mask = query_encodings_dict['attention_mask']

        query_input_ids = torch.cat([query_input_ids.new(cat_ids)[:, None], query_input_ids], dim=1)
        query_mask = torch.cat([query_mask.new([1] * len(query_mask))[:, None], query_mask], dim=1)

        response_encodings_dict = self.tokenizer(responses, return_tensors="pt", padding=True)
        response_input_ids = response_encodings_dict['input_ids']
        response_mask = response_encodings_dict['attention_mask']

        return query_input_ids, query_mask, response_input_ids, response_mask


class FixedController:
    def __init__(self, coef):
        self.value = coef

    def update(self, current, n_steps, lower_bound):
        pass


class AdaptiveController:
    def __init__(self, init_coef, target, horizon):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps, lower_bound):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        if lower_bound:
            mult = 1 + proportional_error * n_steps / self.horizon
        else:
            mult = 1 - proportional_error * n_steps / self.horizon
        self.value *= mult


class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: Policy,
                 data_pool: DataPool,
                 score_model: Reward,
                 tree_tokens: List[str],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR,
                 resume: bool):

        self.params = params
        self.policy = policy
        self.data_pool = data_pool
        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.writer = SummaryWriter(log_dir=params.tensorboard_dir)

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveController(self.params.kl_coef, self.params.target_kl, self.params.horizon)
        else:
            self.kl_ctl = FixedController(self.params.kl_coef)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        if self.params.adaptive_entropy:
            self.entropy_ctl = AdaptiveController(self.params.entropy_coef, self.params.target_entropy,
                                                  self.params.horizon)
        else:
            self.entropy_ctl = FixedController(self.params.entropy_coef)

        self.tree_tokens = tree_tokens
        self.best_cat = self.tree_tokens[0]
        self.best_cat_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat)

        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceCollator(tokenizer=policy.tokenizer)

        if resume:
            sample_dataset = SequenceDataset(data_pool=self.data_pool)
            self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                                shuffle=True, drop_last=True, collate_fn=self.seq_collator)
            self.sampler = iter(self.sample_dataloader)

    def sample(self, step):
        if step % self.params.sample_interval != 0:
            return
        log.info(f"[step {step}] Sampling ...")

        concepts, prompts, responses = [], [], []

        for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader),
                                       desc='Sampling from current policy')):
            input_ids, attention_mask, concept, constraints = batch
            use_constraint = random.choices([1, 0], weights=[self.params.hard_prob, 1 - self.params.hard_prob], k=1)[0]
            rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask,
                                          constraints=constraints if use_constraint else None,
                                          max_len=self.params.response_length, top_p=self.params.top_p,
                                          use_control_code=(step > 0))
            prompt, response = rollouts['query/text'], rollouts['response/text']
            concepts.extend(concept)
            prompts.extend(prompt)
            responses.extend(response)

        scores = self.score_model.get_reward(prompts, responses, concepts, f'step{step}')
        self.data_pool.add(prompts=prompts, responses=responses, scores=scores['reward'])

        sample_dataset = SequenceDataset(data_pool=self.data_pool)
        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                            shuffle=True, drop_last=True, collate_fn=self.seq_collator)
        self.sampler = iter(self.sample_dataloader)

    def step(self, step_num):
        step_started_at = time.time()
        self.save(step=step_num)
        self.eval(step=step_num)
        self.sample(step=step_num)

        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.sampler = iter(self.sample_dataloader)
            batch = next(self.sampler)

        self.policy.value_model.train()
        ppo_loss, stats = self.loss(step_num, *batch)
        ppo_loss = ppo_loss / self.params.grad_accum
        ppo_loss.backward()
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.value_model.parameters(), self.params.max_grad_norm)
        if (step_num + 1) % self.params.grad_accum == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()

        for metric in ['kl', 'entropy']:
            self.writer.add_scalar(f'Objective/{metric}', stats[f'objective/{metric}'], step_num)
        for metric in ['lm', 'kl', 'entropy', 'total']:
            self.writer.add_scalar(f'Loss/{metric}', stats[f'loss/{metric}'], step_num)
        self.writer.add_scalar(f'Params/lr', self.optimizer.param_groups[0]['lr'], step_num)
        self.writer.add_scalar(f'Params/kl_coef', self.kl_ctl.value, step_num)
        self.writer.add_scalar(f'Params/entropy_coef', self.entropy_ctl.value, step_num)

        self.kl_ctl.update(stats['objective/kl'], self.params.batch_size, True)
        self.entropy_ctl.update(stats['objective/entropy'], self.params.batch_size, False)

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        log.info(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")

    def loss(self, step, query_input_ids, query_mask, response_input_ids, response_mask):
        outputs = self.policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask,
                                           use_control_code=True)
        lm_loss, logprobs, entropy, logits = outputs['response/lm_loss'], outputs['response/log_prob'], \
                                             outputs['response/entropy'], outputs['response/logits']
        masks = response_mask.to(self.policy.device)

        with torch.no_grad():
            ref_outputs = self.policy.forward_pass(query_input_ids[:, 1:], query_mask[:, 1:],
                                                   response_input_ids, response_mask, use_control_code=False)
            ref_logprobs, ref_logits = ref_outputs['response/log_prob'], ref_outputs['response/logits']

        kl = torch.sum(self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1)), dim=-1)

        loss = reduce_mean(lm_loss + self.kl_ctl.value * kl - self.entropy_ctl.value * entropy, masks)

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        queries, responses = decode(self.policy.tokenizer, query_input_ids, response_input_ids)
        self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1),
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step=step)

        return loss, stats

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(queries[i] + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

    def save(self, step):
        if step < self.params.min_save_step or step > self.params.max_save_step or step % self.params.save_interval != 0:
            return
        torch.save({
            'step': step,
            'value_model': self.policy.value_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'data_pool': self.data_pool.data_to_save(),
        }, f'{self.params.model_dir}/ckp_{step}.pth')
        log.info(f"[step {step}] model checkpoint saved")

    def eval(self, step):
        if step % self.params.eval_interval != 0:
            return
        log.info(f"[step {step}] evaluating ...")

        concepts, prompts, uncons_gens, cons_gens = [], [], [], []
        for i, batch in enumerate(tqdm(self.val_dataloader)):
            input_ids, attention_mask, concept, constraints = batch
            with torch.no_grad():
                uncons_rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask,
                                                     max_len=self.params.response_length, top_p=self.params.top_p,
                                                     use_control_code=(step > 0))

                cons_rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask,
                                                   constraints=constraints,
                                                   max_len=self.params.response_length, top_p=self.params.top_p,
                                                   use_control_code=(step > 0))

                concepts.extend(concept)
                prompts.extend(uncons_rollouts['query/text'])
                uncons_gens.extend(uncons_rollouts['response/text'])
                cons_gens.extend(cons_rollouts['response/text'])

        for eval_name, gens in [('unconstrained', uncons_gens), ('constrained', cons_gens)]:
            print(f"  {eval_name.capitalize()} evaluation: ")
            score_dict = self.score_model.get_reward(prompts, gens, concepts, f'step{step}_eval_{eval_name}')
            for name, scores in score_dict.items():
                metric_score = np.mean(scores)
                print(f"  {name} = {metric_score:+.2f}")
                self.writer.add_scalar(f'{eval_name.capitalize()}_eval/{name}', metric_score, step)


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    log.info(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.resume is None:
        date_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
        args.save_dir = os.path.join(args.output_dir, date_time)
    else:
        sub_dirs = args.resume.split(os.sep)
        date_time, args.output_dir, args.save_dir = sub_dirs[-3], os.sep.join(sub_dirs[:-3]), os.sep.join(sub_dirs[:-2])
    args.reward_dir = os.path.join(args.save_dir, 'reward')
    args.model_dir = os.path.join(args.save_dir, 'model')
    args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard', date_time)
    log.info(f'Write to output directory: {args.save_dir}')

    if args.resume is None:
        for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
            ensure_dir(d)

        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens)]

    log.info(f'Initializing models ...')
    policy_checkpoint = torch.load(args.base_model_checkpoint, map_location='cpu')['policy_model']
    policy = Policy(base_model_name=args.base_model_name, base_model_checkpoint=policy_checkpoint,
                    value_model_name=args.value_model_name, device=device, tree_tokens=tree_tokens, alpha=args.alpha,
                    calibrate=args.gpt3_calibrate, force_eos=args.force_eos)
    reward = Reward(save_path=args.reward_dir, batch_size=args.reward_batch_size, device=num_gpus - 1, params=args)
    data_pool = DataPool(tree_tokens=tree_tokens, n_extra_tokens=args.n_extra_tokens)
    log.info(f'Initialization done!')

    prompt_collator = PromptCollator(tokenizer=policy.tokenizer)
    train_dataset = PromptDataset(path=args.dataset_train, tokenizer=policy.tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=prompt_collator)
    log.info(f'Load train set with {len(train_dataset)} examples')

    val_dataset = PromptDataset(path=args.dataset_val, tokenizer=policy.tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=prompt_collator)
    log.info(f'Load val set with {len(val_dataset)} examples')

    # set up optimizer and scheduler
    optimizer = Adam(policy.value_model.parameters(), lr=args.lr, eps=1e-5)
    args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)

    start = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        start = checkpoint['step']
        policy.value_model.load_state_dict(checkpoint['value_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        data = checkpoint['data_pool']
        data_pool.add(prompts=data['prompts'], responses=data['responses'], scores=data['scores'])

    trainer = ConditionTrainer(params=args, policy=policy, data_pool=data_pool,
                               score_model=reward, tree_tokens=tree_tokens,
                               train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                               optimizer=optimizer, scheduler=scheduler, resume=start > 0)

    for step_num in range(start, args.total_steps):
        trainer.step(step_num)


if __name__ == "__main__":
    main()

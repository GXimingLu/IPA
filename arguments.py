import torch
import argparse
from utils.constants import HOME_PATH


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # dataset
    parser.add_argument(
        '--output-dir', type=str, default=f'{HOME_PATH}/commonGen')
    parser.add_argument(
        '--dataset-train', type=str, default=f'{HOME_PATH}/data/commongen/train.json',
        help='JSON file containing train prompts. Each item contains "prompt", "response".')
    parser.add_argument(
        '--dataset-val', type=str, default=f'{HOME_PATH}/data/commongen/val.json',
        help='JSON file containing dev prompts. Each item contains "prompt", "response".')

    # reward
    parser.add_argument(
        '--n_extra_tokens', type=int, default=5, help='number of reward categorization')
    parser.add_argument(
        '--sample-interval', type=int, default=750, help='step interval to sample from current policy')
    parser.add_argument(
        '--horizon', type=float, default=2500, help='horizon value in adaptive controller')
    parser.add_argument(
        '--reward_batch_size', type=int, default=16, help='batch size')
    parser.add_argument(
        '--binary_coverage', action='store_true', default=False, help='whether to use binary_coverage')

    # KL term
    parser.add_argument(
        '--kl_coef', type=float, default=0.0, help='coefficient for KL term in reward')
    parser.add_argument(
        '--adaptive_kl', action='store_true', default=False, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--target_kl', type=float, default=3, help='target value in adaptive KL controller')
    # entropy term
    parser.add_argument(
        '--entropy_coef', type=float, default=0.0, help='coefficient for entropy term in reward')
    parser.add_argument(
        '--adaptive_entropy', action='store_true', default=False, help='whether to use adaptive entropy controller')
    parser.add_argument(
        '--target_entropy', type=float, default=40, help='target value in adaptive entropy controller')

    # policy
    parser.add_argument(
        '--base_model_name', type=str, default='gpt2-xl', help='language model as the base policy.')
    parser.add_argument(
        '--base_model_checkpoint', type=str, default="PATH_TO_DISTILLED_GPT3", help='base policy initialization')
    parser.add_argument(
        '--value_model_name', type=str, default='gpt2-large', help='language model as the value function.')
    parser.add_argument(
        '--alpha', type=float, default=1.0, help='co-efficient to combine policy and value model.')
    parser.add_argument(
        '--response-length', type=int, default=64, help='number of tokens to generate for each prompt.')
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='temperature for sampling policy.')
    parser.add_argument(
        '--gpt3_calibrate', action='store_true', default=False, help='calibrate to adapt gpt3 logprobs')

    # training
    parser.add_argument(
        '--total-episodes', type=int, default=2000000, help='total number of episodes')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument(
        '--grad_accum', type=int, default=2, help='gradient accumulation steps')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--num_warmup_steps', type=int, default=500, help='number of warmup steps in lr scheduler')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')

    # generation
    parser.add_argument(
        '--num-samples', type=int, default=1, help='number of samples to generate for each prompt.')
    parser.add_argument(
        '--top-p', type=float, default=0.6, help='hyperparameter for nucleus sampling')
    parser.add_argument(
        '--hard_prob', type=float, default=0.75, help='whether to use hard constraint in decoding')
    parser.add_argument(
        '--force_eos', action='store_true', default=False, help='not to generate eos until all constraints satisfied')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=200, help='step interval to print out logs')
    parser.add_argument(
        '--save-interval', type=int, default=500, help='step interval to save model checkpoints')
    parser.add_argument(
        '--min_save_step', type=int, default=8000, help='minimal steps before saving model checkpoints')
    parser.add_argument(
        '--max_save_step', type=int, default=15000, help='maximal steps for saving model checkpoints')
    parser.add_argument(
        '--eval-interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument(
        '--cuda-deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    parser.add_argument(
        '--resume', type=str, default=None, help='directory to resume generation')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args

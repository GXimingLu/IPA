import torch
import torch.nn.functional as F
import json
import numpy as np
from typing import Union, List, Dict
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from lexical_constraints import ConstrainedHypothesis, init_batch
from utils.constants import NEGATIVE_INF
from utils.utils import logits_to_entropy, mask_pad, process_generation
from utils.generation_utils import add_control_code, get_model_output, remove_control_code, get_response_logits


class Policy:
    def __init__(self, base_model_name, base_model_checkpoint, value_model_name, device, tree_tokens,
                 alpha, calibrate, force_eos):
        self.device = device
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.base_model.load_state_dict(base_model_checkpoint)
        self.value_model = GPT2LMHeadModel.from_pretrained(value_model_name)

        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name, pad_token="<|endoftext|>")
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.value_model.config.pad_token_id = self.tokenizer.pad_token_id

        self.tokenizer.add_tokens(tree_tokens, special_tokens=True)

        weights = self.value_model.get_input_embeddings().weight.detach().numpy()
        mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
        new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in tree_tokens])

        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.value_model.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            new_inits = torch.tensor(new_inits)
            self.value_model.get_input_embeddings().weight[-len(tree_tokens):, :] = new_inits

        self.base_model = self.base_model.to(self.device)
        self.base_model.parallelize()
        self.value_model = self.value_model.to(self.device)
        self.value_model.parallelize()

        self.best_cat = tree_tokens[0]
        self.best_cat_id = self.tokenizer.convert_tokens_to_ids(self.best_cat)

        self.alpha = alpha
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.calibrate = calibrate

        self.eos_tokens = None
        if force_eos:
            self.eos_tokens = self.tokenizer.convert_tokens_to_ids(['.', 'Ġ.', '!', 'Ġ!'])

    def sample(self,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               constraints: List[ConstrainedHypothesis] = None,
               max_len: int = 64,
               min_len: int = 16,
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               use_control_code: bool = False) -> Dict[str, Union[torch.Tensor, List[str]]]:

        use_constraints = constraints is not None
        if use_constraints:
            constraints = init_batch([json.loads(x) for x in constraints], self.eos_tokens)

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)

        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        model_kwargs = {'attention_mask': attention_mask}
        batch_size, input_seq_len = input_ids.shape

        value_input_ids, value_attention_mask = add_control_code(input_ids, attention_mask, self.best_cat_id)
        value_model_kwargs = {'attention_mask': value_attention_mask}

        logits_warper = self.base_model._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=1
        )

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        output_logprob = torch.zeros([batch_size, 0], dtype=torch.float, device=self.device)
        output_mask = torch.ones([batch_size, 0], dtype=torch.long, device=self.device)

        self.value_model.eval()
        with torch.no_grad():
            for step in range(max_len):

                outputs, next_token_logits = get_model_output(self.base_model, step, input_ids, attention_mask, model_kwargs)

                # get logit from value model
                if use_control_code:
                    value_outputs, value_next_token_logits = get_model_output(self.value_model, step, value_input_ids,
                                                                              value_attention_mask, value_model_kwargs)
                    if self.calibrate:
                        next_token_logits = F.log_softmax(next_token_logits)
                    next_token_logits = next_token_logits + self.alpha * value_next_token_logits

                if step < min_len:
                    next_token_logits[:, self.base_model.config.eos_token_id] = float('-inf')
                if use_constraints:
                    for i, constraint in enumerate(constraints):
                        for bad_word in constraint.avoid():
                            next_token_logits[i, bad_word] = float('-inf')
                log_prob = F.log_softmax(next_token_logits, dim=-1)

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    next_token_scores = logits_warper(input_ids, next_token_logits)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # finished sentences should have their next token be a padding token
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)

                    # update output mask
                output_mask = torch.cat([output_mask, unfinished_sequences[:, None]], dim=-1)
                # update output log probability
                token_logprob = torch.gather(log_prob, 1, next_tokens[:, None]).squeeze(1)
                token_logprob = token_logprob * unfinished_sequences + NEGATIVE_INF * (1 - unfinished_sequences)
                output_logprob = torch.cat([output_logprob, token_logprob[:, None]], dim=-1)

                # update generated ids, model inputs for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self.base_model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.base_model.config.is_encoder_decoder
                )

                if use_constraints:
                    constraints = [c.advance(t) for c, t in zip(constraints, next_tokens.tolist())]

                if use_control_code:
                    value_input_ids = torch.cat([value_input_ids, next_tokens[:, None]], dim=-1)
                    value_model_kwargs = self.value_model._update_model_kwargs_for_generation(
                        value_outputs, value_model_kwargs, is_encoder_decoder=self.value_model.config.is_encoder_decoder
                    )

                # if eos_token was found in one sentence, set sentence to finished
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.tokenizer.eos_token_id).long())

                if unfinished_sequences.max() == 0:
                    break

        response_ids = input_ids[:, input_seq_len:]
        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                         for output in response_ids]
        response_text = [process_generation(t) for t in response_text]

        prompt_ids = input_ids[:, :input_seq_len]
        if prompts is None:
            prompts = [self.tokenizer.decode(query, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                       for query in prompt_ids]

        return {
            'query/input_ids': prompt_ids,
            'query/text': prompts,
            'query/mask': attention_mask,
            'response/input_ids': response_ids,
            'response/text': response_text,
            'response/mask': output_mask,
            'response/log_prob': output_logprob,
        }

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor,
                     use_control_code: bool = False):

        query_input_ids = query_input_ids.to(self.device)
        query_mask = query_mask.to(self.device)
        response_input_ids = response_input_ids.to(self.device)
        response_mask = response_mask.to(self.device)

        if use_control_code:
            value_query_input_ids, value_query_mask = query_input_ids, query_mask
            query_input_ids, query_mask = remove_control_code(query_input_ids, query_mask)

        logits = get_response_logits(self.base_model, query_input_ids, response_input_ids, query_mask, response_mask)

        if use_control_code:
            value_logits = get_response_logits(self.value_model, value_query_input_ids, response_input_ids,
                                               value_query_mask, response_mask)
            logits = logits + self.alpha * value_logits

        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, response_input_ids[:, :, None]).squeeze(2)
        output_entropy = logits_to_entropy(logits)
        lm_loss = -1. * output_logprob

        return {
            'response/log_prob': mask_pad(output_logprob, response_mask),
            'response/lm_loss': mask_pad(lm_loss, response_mask),
            'response/entropy': mask_pad(output_entropy, response_mask),
            'response/logits': logits,
        }

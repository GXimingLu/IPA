import torch


def add_control_code(input_ids, attention_mask, control_code):
    input_ids = torch.cat([input_ids.new([control_code] * len(input_ids))[:, None], input_ids], dim=1)
    attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
    return input_ids, attention_mask


def remove_control_code(input_ids, attention_mask):
    input_ids, attention_mask = input_ids[:, 1:], attention_mask[:, 1:]
    return input_ids, attention_mask


def get_model_output(model, step, input_ids, attention_mask, model_kwargs):
    # prepare model inputs
    batch_size, _ = input_ids.shape
    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )

    # in the first decoding step, we want to use the 'real' last position for each sentence
    if step == 0:
        last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
        next_token_logits = outputs.logits[range(batch_size), last_non_masked_idx, :]
    else:
        next_token_logits = outputs.logits[:, -1, :]

    return outputs, next_token_logits


def get_response_logits(model, query_input_ids, response_input_ids, query_mask, response_mask):
    batch_size, query_seq_len = query_input_ids.shape
    input_ids = torch.cat([query_input_ids, response_input_ids], dim=-1)
    model_kwargs = {'attention_mask': torch.cat([query_mask, response_mask], dim=-1)}
    model_inputs =model.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # forward pass to get next token
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    # get the first logit
    query_logits = outputs.logits[:, :query_seq_len, :]
    last_non_masked_idx = torch.sum(query_mask, dim=1) - 1
    first_logits = query_logits[range(batch_size), last_non_masked_idx, :]
    # get the second to last logit
    response_logits = outputs.logits[:, query_seq_len:-1, :]
    logits = torch.cat([first_logits[:, None], response_logits], dim=1)

    return logits


def decode(tokenizer, query_input_ids, response_input_ids=None):
    query = [tokenizer.decode(p, skip_special_tokens=True, clean_up_tokenization_spaces=True)
             for p in query_input_ids]

    if response_input_ids is None:
        return query

    response = [tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for r in response_input_ids]
    return query, response


def get_embedding(tokenizer, model, texts, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings

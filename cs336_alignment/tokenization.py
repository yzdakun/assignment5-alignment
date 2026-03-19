from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
   """Tokenize prompt/output pairs and construct response masks.

   Suggested implementation flow:
   1. Tokenize `prompt_strs` and `output_strs` separately so you can track
      where the response starts for each example.
   2. For each example, concatenate the prompt tokens and output tokens into
      one full sequence.
   3. Pad the batch of full sequences to a common length so they can be
      stacked into a single tensor.
   4. Build language-model training pairs from the padded full sequences:
      `input_ids` is the sequence without the last token, and `labels` is the
      sequence without the first token.
   5. Construct `response_mask` with the same shape as `labels`. Mark only
      those label positions that correspond to response tokens as True; prompt
      and padding positions should remain False.

   Important alignment detail:
   If a prompt has token length P, the first response token appears at index
   P - 1 in `labels` because of the one-token shift used to form
   `(input_ids, labels)`.
   """
   if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

   prompt_token_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
   output_token_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

   pad_token_id = tokenizer.pad_token_id
   if pad_token_id is None:
      if tokenizer.eos_token_id is None:
         raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
      pad_token_id = tokenizer.eos_token_id

   full_sequences = []
   full_response_masks = []
   max_full_len = 0

   for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids, strict=True):
      full_ids = list(prompt_ids) + list(output_ids)
      response_mask = [False] * len(prompt_ids) + [True] * len(output_ids)

      full_sequences.append(full_ids)
      full_response_masks.append(response_mask)
      max_full_len = max(max_full_len, len(full_ids))

   padded_full_sequences = []
   padded_full_response_masks = []
   for full_ids, response_mask in zip(full_sequences, full_response_masks, strict=True):
      pad_len = max_full_len - len(full_ids)
      padded_full_sequences.append(full_ids + [pad_token_id] * pad_len)
      padded_full_response_masks.append(response_mask + [False] * pad_len)

   full_sequence_tensor = torch.tensor(padded_full_sequences, dtype=torch.long)
   full_response_mask_tensor = torch.tensor(padded_full_response_masks, dtype=torch.bool)

   return {
      "input_ids": full_sequence_tensor[:, :-1],
      "labels": full_sequence_tensor[:, 1:],
      "response_mask": full_response_mask_tensor[:, 1:],
   }

   

   raise NotImplementedError

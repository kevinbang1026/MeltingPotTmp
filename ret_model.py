import torch
import os
import re
import json
import pickle
import logging
import requests
import subprocess
import transformers
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from glob import glob
from copy import deepcopy
from torch import Tensor as T, nn
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, BatchSampler
from transformers import AutoTokenizer, AutoModel

pretrained_model_name = 'intfloat/multilingual-e5-base'
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()

class MeltingPotEncoder(torch.nn.Module):
    def __init__(self):
        super(MeltingPotEncoder, self).__init__()
        self.embedding_model_name = pretrained_model_name
        self.passage_encoder = AutoModel.from_pretrained(self.embedding_model_name)
        self.query_encoder = AutoModel.from_pretrained(self.embedding_model_name)

        self.dropout = nn.Dropout(0.1)

        self.emb_sz = (
            self.passage_encoder.pooler.dense.out_features
        )  # get cls token dim

    def forward(
          self,
          input_id,
          attn_mask,
          type = "passage"
        ) :
        assert type in (
            "passage",
            "query",
        ), "type should be either 'passage' or 'query'"

        if type == "passage":
            output = self.dropout(
                self.passage_encoder(input_ids=input_id, attention_mask=attn_mask)
                  .last_hidden_state
            )
        else:
            output = self.dropout(
                self.query_encoder(input_ids=input_id, attention_mask=attn_mask)
                  .last_hidden_state
            )

        return torch.stack([output[i][0] for i in range(len(output))])


    def checkpoint(self, model_ckpt_path):
        torch.save(deepcopy(self.state_dict()), model_ckpt_path)
        logger.debug(f"model self.state_dict saved to {model_ckpt_path}")

    def load(self, model_ckpt_path):
        with open(model_ckpt_path, "rb") as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)
        logger.debug(f"model self.state_dict loaded from {model_ckpt_path}")

class MeltingPotRetriever:
    def __init__(self, model, device='cuda:0'):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def get_embedding(self, content: str, emb_type: str = "query"):
        self.model.eval()
        tok = self.tokenizer(content, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
          out = self.model(
                  T(tok['input_ids']).to(self.device).long(),
                  T(tok['attention_mask']).to(self.device).long(),
                  emb_type
                )

        return out
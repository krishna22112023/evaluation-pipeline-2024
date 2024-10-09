import torch
from torch.nn import functional as F
import yaml
import json
from tqdm import tqdm
from lm_eval.api.model import LM
from typing import List, Tuple
from transformers import AutoTokenizer,AutoModelForMaskedLM, AutoModelForCausalLM, AutoConfig
from omegaconf import OmegaConf
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
import logging
from minicons.scorer import MaskedLMScorer
import sys
from pathlib import Path
base = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(base)
logger = logging.getLogger(__name__)

@register_model("roberta-custom")
class RobertaLM(LM):
    def __init__(self, config_path,batch_size=None,device=None,max_batch_size=None,image_src=None):
        super().__init__()
        self.args = OmegaConf.load(config_path)            
        self._max_length = 512   
        self.tokenizer = self.load_tokenizer()   
        self.vocab_size = self.tokenizer.vocab_size
        if self.args.eval.model_location == "local":
            self.model = self.load_model()
        else:
            self.model = self.load_hf_model()
        self.batch_size = 1
        self.device = self.config["device"]
    
    def load_tokenizer(self):
        if self.args.eval.tokenizer_type == "pretrained":
            return AutoTokenizer.from_pretrained(f"{base}/models/{self.args.general.exp_name}/tokenizer",trust_remote_code=True,use_fast=True)
        elif self.args.eval.tokenizer_type == "pretrained_hf":
            if self.args.eval.model_name_or_path is not None:
                return AutoTokenizer.from_pretrained(self.args.eval.hf_model_name)
            else:
                raise ValueError("Invalid/Missing hf tokenizer in config")
    
    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            pass
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding
    
    def load_config(self,config_path):
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    
    def load_hf_model(self) -> torch.nn.Module:
        if self.args.WML.model_type == "CLM":
            base_model = AutoModelForCausalLM.from_pretrained(self.args.WML.hf_model_name, trust_remote_code=True)
            for param in base_model.parameters():
                param.requires_grad = True
            logger.info("Loaded a CLM huggingface model")
            return torch.compile(base_model.to(self.device))
        elif self.args.WML.model_type == "MLM":
            base_model = AutoModelForMaskedLM.from_pretrained(self.args.WML.hf_model_name, trust_remote_code=True)
            for param in base_model.parameters():
                param.requires_grad = True
            logger.info("Loaded a MLM huggingface model")
            return torch.compile(base_model.to(self.device))
        else:
            raise Exception(f"Error: Invalid model type {self.args.WML.model_type}")
        
    def load_model(self):
        checkpoint = torch.load(f"{base}/{self.args.eval.model_name_or_path}", map_location=self.args.WML.device)
        state_dict = checkpoint['model']
        with open(f"{base}/{self.args.eval.model_config_path}", 'r') as json_file:
            config_dict = json.load(json_file)
        config = AutoConfig.for_model(**config_dict)
        logger.info(f"loaded config {config}")
        if self.args.WML.model_type == "MLM":
            peer_model = AutoModelForMaskedLM.from_config(config)
        elif self.args.WML.model_type == "CLM":
            peer_model = AutoModelForCausalLM.from_config(config)
        peer_model.load_state_dict(state_dict)
        peer_model.to(self.config["device"])
        num_params = sum(torch.sum(p != 0).item() for p in peer_model.parameters() if p.requires_grad)
        logger.info(f"loaded a local peer model with {num_params} parameters")
        return peer_model

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        TODO: Currently only works for relative position encoded Seq2Seq models.
        """
        if self._max_length is not None:
            return self._max_length
        return self._DEFAULT_MAX_LENGTH

    def _model_call(self, inps, attn_mask=None, labels=None):
        return self.model(inps, labels=labels)

    def _model_generate(self, inputs, max_tokens, stop=None):
        raise NotImplementedError("Masked LMs are not well-suited to generating sequences.")
    
    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self.batch_schedule)
        if sched in self.batch_sizes:
            return self.batch_sizes[sched]
        if (len(self.batch_sizes) > 1) and (
            self.batch_sizes[sched - 1] == self.max_batch_size
        ):
            # if previous batch size is already maximal, skip recomputation
            self.batch_sizes[sched] = self.max_batch_size
            return self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
        )
        self.batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self.batch_sizes[sched]}")
        return self.batch_sizes[sched]

    def loglikelihood(self, requests, disable_tqdm=False):
        """
        Returns *pseudo*-loglikelihoods, as described in Salazar et al. (2020).
        """
        
        # assert getattr(self.config, "model_type") in MODEL_FOR_MASKED_LM_MAPPING_NAMES, \
        #     "Used `--model hf-mlm`, but model doesn't have an AutoModelForMaskedLM implementation!"

        def _collate(req: Tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        scores = []
                
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )

        if self.tokenizer.mask_token_id is None:
            self.tokenizer.mask_token_id = self.model.config.mask_token_id
            self.tokenizer_pad_token_id = self.model.config.pad_token_id

        scorer = MaskedLMScorer(self.model, self.device, tokenizer=self.tokenizer)

        for chunk in chunks:
            # chunk_items = zip(*chunk)
            context, continuation = zip(*chunk)
            context = [
                f"{self.tokenizer.eos_token}" if len(text) == 0 else text for text in context
            ]

            chunk_logprobs = scorer.conditional_score(context, continuation)
            batch_scores = [(p, False) for p in chunk_logprobs]
            scores.extend(batch_scores)
            pbar.update(len(chunk))
        pbar.close()

        return scores
    
    def generate_until(self, requests) -> List[str]:
        raise NotImplementedError("Masked LMs are not well-suited to generating sequences.")
    
    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        raise NotImplementedError("Masked LMs are not well-suited to generating sequences.")
import coremltools as ct
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from typing import Iterable, List, Optional, Tuple, Union

from lm_eval.base import BaseLM

TokenSequence = List[int]


class HuggingFaceCoreMLLM(BaseLM):
    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model.
    _DEFAULT_MAX_LENGTH: int = 128

    def __init__(
        self,
        pretrained: str,
        tokenizer: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 1,
        max_gen_toks: Optional[int] = 256,
        max_length: Optional[int] = None,
    ):
        """Initializes a HuggingFace Core ML model and a tokenizer for evaluation.
        Args:
            pretrained (str):
                Path to an mlpackage bundle in the filesystem.
                Hugging Face Hub models are not supported yet.
        """
        super().__init__()

        assert isinstance(pretrained, str)
        assert int(batch_size) == 1

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length

        # TODO: snapshot_download from the hub if it's remote
        # TODO: make compute_units configurable
        self.model = ct.models.MLModel(pretrained, compute_units=ct.ComputeUnit.ALL)
        original_pretrained = self.model.user_defined_metadata["co.huggingface.exporters.name"]

        self.tokenizer = self._create_auto_tokenizer(
            pretrained=original_pretrained,
            tokenizer=tokenizer,
        )
        self.tokenizer.model_max_length = self.max_length

    def _create_auto_tokenizer(
        self,
        *,
        pretrained: str,
        tokenizer: Optional[str] = None,
    ) -> transformers.PreTrainedTokenizer:
        """Returns a pre-trained tokenizer from a pre-trained tokenizer configuration."""
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        """
        if self._max_length is not None:
            return self._max_length
        
        # Try to get the sequence length from the model metadata
        spec = self.model.get_spec()
        try:
            # TODO: update for discrete and flexible shapes
            input_feature = next(filter(lambda x: x.name == "input_ids", spec.description.input))
            return input_feature.type.multiArrayType.shape[-1]
        except:
            return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Support other sizes
        return 1

    @property
    def device(self):
        return "cpu"
    
    @property
    def requires_attention(self):
        return "attention_mask" in self.model.input_description

    def tok_encode(self, string: str) -> TokenSequence:
        return self.tokenizer.encode(string)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.tokenizer(
            strings,
            padding=True,
            return_tensors="np",
        )

    def tok_decode(self, tokens: Iterable[int]) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inputs) -> TokenSequence:
        ml_inputs = { "input_ids": inputs.numpy().astype(np.int32) }
        if self.requires_attention:
            ml_inputs["attention_mask"] = np.ones(inputs.shape, np.int32)
        logits = self.model.predict(ml_inputs)["logits"]
        return torch.tensor(logits)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        pass
    
    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        pass
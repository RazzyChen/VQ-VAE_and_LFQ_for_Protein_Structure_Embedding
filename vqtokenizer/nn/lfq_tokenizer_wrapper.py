import torch

from .lfq_backbone import LFQTokenizer


class LFQTokenizerWrapper:
    """
    Wrapper for LFQTokenizer to provide Hugging Face-style tokenizer interface.
    Usage:
        lfq_tokenizer = LFQTokenizer(...)
        wrapper = LFQTokenizerWrapper(lfq_tokenizer)
        out = wrapper(features, return_tensors='pt')
        # out['input_ids']
    """

    def __init__(self, lfq_tokenizer: LFQTokenizer):
        self.lfq_tokenizer = lfq_tokenizer

    def __call__(self, features, return_tensors=None, **kwargs):
        input_ids = self.encode(features)
        output = {"input_ids": input_ids}
        if return_tensors == "pt":
            output["input_ids"] = torch.as_tensor(input_ids)
        return output

    def encode(self, features):
        """
        features: torch.Tensor, shape [N, input_dim]
        Returns: list of token ids
        """
        self.lfq_tokenizer.eval()
        with torch.no_grad():
            indices = self.lfq_tokenizer.encode(features)
        if isinstance(indices, torch.Tensor):
            return indices.cpu().tolist()
        return indices

    def decode(self, input_ids):
        """
        input_ids: list or tensor of indices
        Returns: reconstructed features (torch.Tensor)
        """
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        self.lfq_tokenizer.eval()
        with torch.no_grad():
            features = self.lfq_tokenizer.decode(input_ids)
        return features

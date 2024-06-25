from typing import Any, Dict, List
import hydra

from torch.utils.data import DataLoader

from roleplay.common.model import Model


class Dataset:
    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        self.cfg = cfg
        self.dataset = None
        self._get_dataset()

    @staticmethod
    def get_dataset(cfg):
        return hydra.utils.instantiate(cfg.type, cfg)

    def _get_dataset(self):
        raise NotImplementedError("method '_get_dataset' is not implemented")

    def get_dataloader(self, dataloader_cfg: List[Dict[str, Any]], tokenizer, input_key):
        return DataLoader(
            self.dataset,
            batch_size=dataloader_cfg.get("batch_size"),
            num_workers=dataloader_cfg.get("num_workers", 2),
            shuffle=dataloader_cfg.get("shuffle"),
            collate_fn=lambda data: Model.collate_tokenize(data=data, tokenizer=tokenizer, input_key=input_key),
        )

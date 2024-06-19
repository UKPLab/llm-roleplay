from typing import Any, Dict, List

from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader

from roleplay.common.model import Model


class Dataset:
    @staticmethod
    def get_dataset(cfg: List[Dict[str, Any]]):
        if cfg.get("name"):
            return load_dataset(
                cfg.name,
                cfg.subset,
                split=cfg.get("split"),
            )
        else:
            return HFDataset.from_dict(dict(cfg))

    @staticmethod
    def get_datasets(cfg: List[Dict[str, Any]]) -> List[Any]:
        datasets: List[Any] = []
        for dataset_cfg in cfg:
            datasets.append(Dataset.get_dataset(dataset_cfg))
        return datasets

    @staticmethod
    def get_dataloader(
        dataset,
        tokenizer,
        dataloader_cfg: List[Dict[str, Any]],
        dataset_cfg: List[Dict[str, Any]],
    ):
        return DataLoader(
            dataset,
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
            shuffle=dataloader_cfg.shuffle,
            collate_fn=lambda data: Model.collate_tokenize(data, tokenizer, dataset_cfg),
        )

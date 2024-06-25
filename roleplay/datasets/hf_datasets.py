from typing import Any, Dict, List

from datasets import load_dataset

from roleplay.common.dataset import Dataset


class HFDatasets(Dataset):
    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        super().__init__(cfg)

    def _get_dataset(self):
        self.dataset = load_dataset(
            self.cfg.name,
            self.cfg.get("subset"),
            split=self.cfg.get("split"),
        )

from typing import Any, Dict, List

from datasets import Dataset as HFDataset
from urartu.common.dataset import Dataset


class HFDatasets(Dataset):
    def __init__(self, cfg: List[Dict[str, Any]]) -> None:
        super().__init__(cfg)

    def _get_dataset(self):
        self.dataset = HFDataset.from_dict(dict(self.cfg.data))

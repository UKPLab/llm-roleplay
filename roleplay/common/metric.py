from typing import Any, Dict, List

import evaluate


class Metric:
    @staticmethod
    def get_metric(cfg: List[Dict[str, Any]]):
        return evaluate.load(cfg.name)

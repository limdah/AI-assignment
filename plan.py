# plan.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Plan:
    model: str = "logistic_regression"
    drop_columns: List[str] = field(default_factory=list)
    keep_only_columns: Optional[List[str]] = None

    # Defaults (used by pipeline)
    numeric_impute: str = "median"
    categorical_impute: str = "most_frequent"
    encoding: str = "one_hot"

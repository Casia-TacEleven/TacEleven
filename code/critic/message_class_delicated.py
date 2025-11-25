from dataclasses import dataclass
import numpy as np


@dataclass
class DecisionTask:
    history: list
    condition: str
    time_hist: list
    time_pred: list
    description: dict
    label: list
    index: int
    step: int

@dataclass
class ReviewTask:
    session_id: str
    path: str
    decision_task_dict: dict
    decision: list
    decision_sketch: dict
    decision_candidate: dict

@dataclass
class ReviewResult:
    session_id: str
    event_advised: dict
    approved: bool

@dataclass
class DecisionResult:
    task_dict: dict
    decision: list
    event_advised: dict


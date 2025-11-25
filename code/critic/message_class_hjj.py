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

# -- added
@dataclass
class ActionChoice:
    session_id: str
    event_advised: dict
    approved: bool
    needs_resampling: bool  # 重采样
    num_samples: int

@dataclass
class ResamplingTask:
    session_id: str
    event_advised: dict
    base_decision_task: dict
    num_samples: int

@dataclass
class ResamplingResult:
    session_id: str
    path: str
    event_advised: dict
    base_decision_task: dict
    resampled_predictions: dict
    resampled_sketches: dict

@dataclass
class FinalReviewResult:
    session_id: str
    event_advised: dict
    approved: bool
    needs_resampling: bool
    selected_prediction: dict
    resampled_sketches: dict
    selected_option_index: int

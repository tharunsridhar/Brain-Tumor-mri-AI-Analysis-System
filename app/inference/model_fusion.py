from __future__ import annotations

import numpy as np

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
DEFAULT_WEIGHTS = {
    'EfficientNetV2-S': 1.2,
    'MobileNetV3': 1.0,
    'ConvNeXt Tiny': 1.1,
}


def normalize_scores(scores):
    arr = np.asarray(scores, dtype=np.float32)
    total = float(arr.sum())
    if total <= 0:
        return np.zeros_like(arr)
    return arr / total


def fuse_model_outputs(model_outputs, weights=None, class_names=None):
    weights = weights or DEFAULT_WEIGHTS
    class_names = class_names or CLASS_NAMES
    weighted = np.zeros(len(class_names), dtype=np.float32)
    votes = {}
    total_weight = 0.0
    pretty = {}
    for name, scores in model_outputs.items():
        probs = normalize_scores(scores)
        pretty[name] = {label: round(float(value), 4) for label, value in zip(class_names, probs)}
        weighted += probs * float(weights.get(name, 1.0))
        total_weight += float(weights.get(name, 1.0))
        votes[name] = class_names[int(np.argmax(probs))]
    weighted = weighted / max(total_weight, 1.0)
    ranked = np.argsort(weighted)[::-1]
    top_idx = int(ranked[0])
    second = float(weighted[int(ranked[1])]) if len(ranked) > 1 else 0.0
    agreement = sum(1 for vote in votes.values() if vote == class_names[top_idx]) / max(len(votes), 1)
    return {
        'final_class': class_names[top_idx],
        'fused_confidence': round(float(weighted[top_idx]), 4),
        'agreement_score': round(float(agreement), 4),
        'margin': round(float(weighted[top_idx] - second), 4),
        'model_votes': votes,
        'class_scores': {label: round(float(score), 4) for label, score in zip(class_names, weighted)},
        'per_model_scores': pretty,
    }


def run_demo():
    sample = {
        'EfficientNetV2-S': [0.72, 0.12, 0.08, 0.08],
        'MobileNetV3': [0.61, 0.17, 0.10, 0.12],
        'ConvNeXt Tiny': [0.58, 0.18, 0.14, 0.10],
    }
    return fuse_model_outputs(sample)


if __name__ == '__main__':
    print(run_demo())


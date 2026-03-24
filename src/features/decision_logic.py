from __future__ import annotations


def make_decision(fused_confidence, agreement_score, margin, low_margin_threshold=0.08):
    uncertainty_flag = bool(margin < low_margin_threshold)
    if fused_confidence < 0.55 or agreement_score < 0.5:
        decision = 'Escalate for specialist review'
    elif uncertainty_flag:
        decision = 'Recheck with segmentation and report caution'
    else:
        decision = 'Accept fused output'
    return {
        'fused_confidence': round(float(fused_confidence), 4),
        'agreement_score': round(float(agreement_score), 4),
        'margin': round(float(margin), 4),
        'uncertainty_flag': uncertainty_flag,
        'decision_logic': decision,
    }


def run_demo():
    return make_decision(0.781, 1.0, 0.233)


if __name__ == '__main__':
    print(run_demo())

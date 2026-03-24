from __future__ import annotations

import importlib
import traceback

FEATURE_MODULES = [
    ('6', 'app.analytics.scan_quality', 'run_demo'),
    ('7', 'app.inference.model_fusion', 'run_demo'),
    ('8', 'app.analytics.decision_logic', 'run_demo'),
    ('9', 'app.analytics.tumor_size', 'run_demo'),
    ('10', 'app.analytics.shape_irregularity', 'run_demo'),
    ('11', 'app.analytics.explainability_overlap', 'run_demo'),
    ('12', 'app.analytics.risk_scoring', 'run_demo'),
    ('13', 'app.analytics.prior_case_comparison', 'run_demo'),
    ('14', 'app.inference.report_builder', 'run_demo'),
    ('15', 'app.analytics.history_dashboard', 'run_demo'),
]


def main():
    passed = 0
    for number, module_name, func_name in FEATURE_MODULES:
        try:
            module = importlib.import_module(module_name)
            result = getattr(module, func_name)()
            print(f'[PASS] Feature {number}: {module_name}')
            print(result)
            print('-' * 72)
            passed += 1
        except Exception:
            print(f'[FAIL] Feature {number}: {module_name}')
            print(traceback.format_exc())
            print('-' * 72)
    print(f'Summary: {passed}/{len(FEATURE_MODULES)} feature modules passed')


if __name__ == '__main__':
    main()



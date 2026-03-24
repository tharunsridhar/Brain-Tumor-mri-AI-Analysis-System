from app.analytics.explainability_overlap import run_demo

result = run_demo()
assert 'overlap_score' in result
assert 'explainability_consistency' in result
print('feature_11_explainability_overlap test passed')
print(result)


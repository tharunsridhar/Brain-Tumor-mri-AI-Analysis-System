from app.analytics.decision_logic import run_demo

result = run_demo()
assert 'decision_logic' in result
assert 'uncertainty_flag' in result
print('feature_08_decision_logic test passed')
print(result)


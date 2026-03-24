from app.analytics.risk_scoring import run_demo

result = run_demo()
assert 'severity' in result
assert 'reliability_score' in result
print('feature_12_risk_scoring test passed')
print(result)


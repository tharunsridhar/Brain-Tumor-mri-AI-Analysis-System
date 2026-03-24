from app.analytics.history_dashboard import run_demo

result = run_demo()
assert 'summary' in result
assert 'latest' in result
print('feature_15_history_dashboard test passed')
print(result)


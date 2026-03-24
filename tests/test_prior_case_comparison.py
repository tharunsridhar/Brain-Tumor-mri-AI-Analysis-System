from app.analytics.prior_case_comparison import run_demo

result = run_demo()
assert 'prior_available' in result
assert 'progression_flag' in result
print('feature_13_prior_case_comparison test passed')
print(result)


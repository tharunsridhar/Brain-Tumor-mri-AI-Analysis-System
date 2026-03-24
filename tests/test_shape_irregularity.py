from app.analytics.shape_irregularity import run_demo

result = run_demo()
assert 'shape' in result
assert 'mass_effect' in result
print('feature_10_shape_irregularity test passed')
print(result)


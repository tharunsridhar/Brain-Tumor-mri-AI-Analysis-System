from app.analytics.tumor_size import run_demo

result = run_demo()
assert 'area_cm2' in result
assert 'bbox' in result
print('feature_09_tumor_size test passed')
print(result)


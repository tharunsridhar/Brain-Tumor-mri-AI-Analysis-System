from app.analytics.scan_quality import run_demo

result = run_demo()
assert 'quality_score' in result
assert 'usable_for_analysis' in result
print('feature_06_scan_quality test passed')
print(result)


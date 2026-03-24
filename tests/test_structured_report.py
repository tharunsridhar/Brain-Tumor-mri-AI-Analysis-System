from app.inference.report_builder import run_demo

result = run_demo()
assert '1. CLINICAL INDICATION' in result
assert '6. RECOMMENDATIONS' in result
print('feature_14_structured_report test passed')
print(result)


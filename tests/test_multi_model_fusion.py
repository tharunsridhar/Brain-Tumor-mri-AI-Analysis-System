from app.inference.model_fusion import run_demo

result = run_demo()
assert 'final_class' in result
assert 'agreement_score' in result
print('feature_07_multi_model_fusion test passed')
print(result)


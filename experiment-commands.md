## 1) LlaMa 3, ML1M, open-gen
### Seed: 121958
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 121958 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 671155
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 671155 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 131932
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 131932 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

## 2) LlaMa 3, Amazon, open-gen
### Seed: 121958
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 121958 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets amazon \
   --baseline_prompts both
```

### Seed: 671155
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 671155 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets amazon \
   --baseline_prompts both
```

### Seed: 131932
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 131932 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets amazon \
   --baseline_prompts both
```

## 3) LlaMa 3, ML1M, Re-ranking
### Seed: 121958
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 121958 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 671155
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 671155 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 131932
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 131932 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```

## 4) LlaMa 3, Amazon, Re-ranking
### Seed: 121958
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 121958 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets amazon \
   --baseline_prompts both
```

### Seed: 671155
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 671155 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets amazon \
   --baseline_prompts both
```

### Seed: 131932
```python
python scripts/run_facter.py \
   --base_model llama3 \
   --seeds 131932 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets amazon \
   --baseline_prompts both
```

## 5) LlaMa 2, ML1M, open-gen
### Seed: 121958
```python
python scripts/run_facter.py \
   --base_model llama2 \
   --seeds 121958 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 671155
```python
python scripts/run_facter.py \
   --base_model llama2 \
   --seeds 671155 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 131932
```python
python scripts/run_facter.py \
   --base_model llama2 \
   --seeds 131932 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

## 6) LlaMa 2, ML1M, Re-ranking
### Seed: 121958
```python
python scripts/run_facter.py \
   --base_model llama2 \
   --seeds 121958 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 671155
```python
python scripts/run_facter.py \
   --base_model llama2 \
   --seeds 671155 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 131932
```python
python scripts/run_facter.py \
   --base_model llama2 \
   --seeds 131932 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```

## 7) Mistral, ML1M, open-gen
### Seed: 121958
```python
python scripts/run_facter.py \
   --base_model mistral \
   --seeds 121958 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 671155
```python
python scripts/run_facter.py \
   --base_model mistral \
   --seeds 671155 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 131932
```python
python scripts/run_facter.py \
   --base_model mistral \
   --seeds 131932 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode open \
   --datasets ml-1m \
   --baseline_prompts both
```

## 8) Mistral, ML1M, Re-ranking
### Seed: 121958
```python
python scripts/run_facter.py \
   --base_model mistral \
   --seeds 121958 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 671155
```python
python scripts/run_facter.py \
   --base_model mistral \
   --seeds 671155 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```

### Seed: 131932
```python
python scripts/run_facter.py \
   --base_model mistral \
   --seeds 131932 \
   --protected_attrs gender,age,occupation \
   --max_iteration 3 \
   --progress \
   --predict_mode rank \
   --datasets ml-1m \
   --baseline_prompts both
```
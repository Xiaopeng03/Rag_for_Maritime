import json
import pandas as pd

with open('maritimebench_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)  # 返回字典或列表
    print(type(data))

df = pd.DataFrame([data])
df.to_json('output.json', force_ascii=False, orient='records', indent=2)
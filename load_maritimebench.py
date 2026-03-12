from modelscope import MsDataset
import json

# 加载数据集
print("📥 正在加载 MaritimeBench 数据集...")
ds_dict = MsDataset.load('HiDolphin/MaritimeBench', subset_name='default', split='test')

print(f"✅ 数据集加载完成，共 {len(ds_dict)} 条数据")
print(f"📊 数据特征: {ds_dict.features}")

# 转换为列表
data_list = []
for i, item in enumerate(ds_dict):
    data_list.append({
        'index': i,
        'question': item['question'],
        'answer': item['answer'],
        'A': item['A'],
        'B': item['B'],
        'C': item['C'],
        'D': item['D']
    })

# 保存为JSON文件
output_file = 'maritimebench_test.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"💾 数据已保存到: {output_file}")
print(f"📝 示例数据:")
print(json.dumps(data_list[0], ensure_ascii=False, indent=2))
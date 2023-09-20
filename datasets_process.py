# %%
import os
from transformers import BertTokenizer, BertModel

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# %%
from datasets import load_dataset
#加载数据
#注意：如果你的网络不允许你执行这段的代码，则直接运行【从磁盘加载数据】即可，我已经给你准备了本地化的数据文件
#转载自seamew/ChnSentiCorp
dataset = load_dataset(path="lansinuote/ChnSentiCorp")

dataset

# %%
#保存数据集到磁盘
#注意：运行这段代码要确保【加载数据】运行是正常的，否则直接运行【从磁盘加载数据】即可
dataset.save_to_disk(dataset_dict_path='./data/ChnSentiCorp')

# %%
#从磁盘加载数据
from datasets import load_from_disk

dataset = load_from_disk('./data/ChnSentiCorp')

dataset

# %%
#取出训练集
dataset = dataset['train']

dataset

# %%
#sort

#未排序的label是乱序的
print(dataset['label'][:10])

#排序之后label有序了
sorted_dataset = dataset.sort('label')
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])

# %%
#shuffle

#打乱顺序
shuffled_dataset = sorted_dataset.shuffle(seed=42)
shuffled_dataset['label'][:10]

# %%
#select
dataset.select([0,10,20,30,40,50])

# %%
#filter
def f(data):
    return data['text'].startswith('选择')

start_with_ar =  dataset.filter(f)
len(start_with_ar),start_with_ar['text']

# %%
#train_test_split, 切分训练集和测试集
dataset.train_test_split(test_size = 0.1)

# %%
#shard
#把数据切分到4个桶中,均匀分配
dataset.shard(num_shards=4,index=0)

# %%
#rename_column
dataset.rename_column('text', 'textA')

# %%
#remove_columns
dataset.remove_columns(['text'])

# %%
#set_format
dataset.set_format(type='torch',columns=['label'])
dataset[0]

# %%
#map
def f(data):
    data['text'] = 'my sentence:' + data['text']
    return data

dataset_map = dataset.map(f)

dataset_map['text'][:5]

# %%
#第三章/导出为csv格式
dataset = load_dataset(path='lansinuote/ChnSentiCorp',split='train')
dataset.to_csv(path_or_buf='./data/ChnSentiCorp.csv')

#加载csv格式数据
csv_dataset = load_dataset(path='csv',
                           data_files='./data/ChnSentiCorp.csv',
                           split='train')
csv_dataset[20]

# %%
print(os.getcwd())
# %%
import os
from transformers import BertTokenizer, BertModel

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# %%
from datasets import list_metrics

# 列出评价指标
metrics_list = list_metrics()
len(metrics_list), metrics_list

# %%
from datasets import load_metric

#加载一个评价指标
metric = load_metric('glue', 'mrpc')

print(metric.inputs_description)

# %%
predictions = [0, 1, 0]
references = [0, 1, 1]

final_score = metric.compute(predictions=predictions, references=references)

final_score
# %%

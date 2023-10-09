# %%
import os
from transformers import pipeline

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# %%
#文本分类
classifier = pipeline("sentiment-analysis")

result = classifier("I hate you")[0]
print(result)

result = classifier("I love you")[0]
print(result)

# %%
question_answerer = pipeline("question-answering")

context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a 
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune 
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
"""

result = question_answerer(question="What is extractive question answering?",
                           context=context)
print(result)

result = question_answerer(
    question="What is a good example of a question answering dataset?",
    context=context)

print(result)

# %%
#完形填空
unmasker = pipeline("fill-mask")

from pprint import pprint

sentence = 'HuggingFace is creating a <mask> that the community uses to solve NLP tasks.'

unmasker(sentence)

# %%

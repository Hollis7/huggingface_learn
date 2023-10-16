# %%
import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# %%
import torch
print(torch.__version__)
print(torch.version.cuda)
# %%
#快速演示
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
print('device=', device)

from transformers import BertModel

#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')
#需要移动到cuda上
pretrained.to(device)

#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)


#定义下游任务模型
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


model = Model()
#同样要移动到cuda
model.to(device)

#虚拟一批数据,需要把所有的数据都移动到cuda上
input_ids = torch.ones(16, 100).long().to(device)
attention_mask = torch.ones(16, 100).long().to(device)
token_type_ids = torch.ones(16, 100).long().to(device)
labels = torch.ones(16).long().to(device)

#试算
model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape

#后面的计算和中文分类完全一样，只是放在了cuda上计算

# %%

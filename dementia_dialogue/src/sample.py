import torch
from transformers.modeling_bert import BertForMaskedLM
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer

tokenizer = BertJapaneseTokenizer.from_pretrained("bert-base-japanese-whole-word-masking")
model = BertForMaskedLM.from_pretrained("bert-base-japanese-whole-word-masking")
model.eval()

input_ids = tokenizer.encode(f"""
    山田さんが{tokenizer.mask_token}を見たのはこれが初めてでした。巨大だった。
    """, return_tensors="pt")

masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

result = model(input_ids)
result = result[0][:, masked_index].topk(5).indices.tolist()[0]
for r in result:
    output = input_ids[0].tolist()
    output[masked_index] = r
    print(tokenizer.decode(output))
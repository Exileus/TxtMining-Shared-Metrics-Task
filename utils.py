import torch
import torch.nn.functional as F

def similarity(embeddings_1, embeddings_2):
    return torch.diagonal(torch.matmul(
        embeddings_1, embeddings_2.transpose(0, 1)
    ))


def get_embeddings(sentences, tokenizer, model, pool_norm=True):
    encoded_input = tokenizer(sentences, padding=True, max_length=1024, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    return F.normalize(model_output.pooler_output)
    
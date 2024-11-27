from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.utils import is_torch_npu_available

from utils import get_docs_data, get_query_data, save_to_json, json_to_dict, save_to_pkl, read_pkl
import heapq
import json
from tqdm import tqdm
from scipy.sparse import csr_matrix
import pickle
from GTEEmbedding import GTEEmbedding





    
def get_dense_embedding_matrix(embedding, pkl_save_path):
    dense_embedding_list = [v["embedding"]["dense_embeddings"] for v in embedding.values()]
    dense_embedding_matrix = torch.tensor(dense_embedding_list).squeeze(1)

    with open(pkl_save_path, 'wb') as f:
        pickle.dump(dense_embedding_matrix, f)


def get_one_hot_matrix(token_weights_list, vocab_dict_file_path="/home/ruzhi/work/mgte/data/vocab_dict.json"):
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')
    vocab_size = len(tokenizer)
    # print("词表大小为：", vocab_size)
    row, col, data = [], [], []
    with open(vocab_dict_file_path,'r',encoding='utf-8') as file:
        vocab_dict = json.load(file)

    count = 0
    for index, token_weight in enumerate(token_weights_list):
        count += 1
        for token, weight in token_weight.items():
            try:
                if str(token) not in vocab_dict:
                    token = "▁" + str(token)
                token_id = vocab_dict[str(token)]
                row.append(index)
                col.append(token_id)
                data.append(weight)

            except Exception as e:
                print(f"Error processing token '{token}': {e}")
                raise  
        
    return  csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(count, vocab_size))



def get_token_weights_matrix(embedding, pkl_save_path):
    token_weights_list = [v["embedding"]["token_weights"][0] for v in embedding.values()]
    token_weights_matrix = get_one_hot_matrix(token_weights_list)
    
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(token_weights_matrix, f)

    

def prepare_embedding(doc_file_path, docs_embedding_file_path, query_file_path, query_embedding_file_path):
    model = GTEEmbedding('Alibaba-NLP/gte-multilingual-base')

    
    docs_dic = get_docs_data(doc_file_path)
    for v in tqdm(docs_dic.values()):
        v["embedding"] = model.encode(v["text"], return_dense = True, return_sparse = True)
    # save_to_pkl(file_path=docs_embedding_file_path, data=docs_dic)
    save_to_json("/home/ruzhi/work/mgte/data/docs_embedding.json", docs_dic)

    
    query_dic = get_query_data(query_file_path)
    for v in tqdm(query_dic.values()):
        v["embedding"] = model.encode(v["query"], return_dense = True, return_sparse = True)
    # save_to_pkl(file_path=query_embedding_file_path, data=query_dic)
    save_to_json("/home/ruzhi/work/mgte/data/query_embedding.json", query_dic)
    

if __name__ == '__main__':
    doc_file_path = "/home/ruzhi/work/mgte/data/corpus.jsonl"
    query_file_path = "/home/ruzhi/work/mgte/data/MLDR/en/MLDR_test.jsonl"
    docs_embedding_save_path = "/home/ruzhi/work/mgte/data/docs_embedding.json"
    query_embedding_sava_path = "/home/ruzhi/work/mgte/data/query_embedding.json"

    # 读取带有embedding的数据
    query_embedding = json_to_dict(query_embedding_sava_path)
    docs_embedding = json_to_dict(docs_embedding_save_path)

    print("读取embedding数据完成!")
    get_dense_embedding_matrix(embedding=query_embedding, pkl_save_path='/home/ruzhi/work/mgte/data/rewrite_query_dense_embedding_matrix.pkl')
    get_token_weights_matrix(query_embedding, '/home/ruzhi/work/mgte/data/rewrite_query_token_weights_matrix.pkl')

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.utils import is_torch_npu_available

from utils import get_docs_data, get_query_data, save_to_json,json_to_dict
import heapq
import json
from tqdm import tqdm
from scipy.sparse import csr_matrix
import pickle
from GTEEmbedding import GTEEmbedding

FILE_PATH_PREFIX = "/home/ruzhi/work/mgte/data/"



def prepare_dense_embedding_matrix(query_embedding, docs_embedding):
    
    def get_dense_embedding_matrix(embedding, pkl_save_path):
        dense_embedding_list = [v["embedding"]["dense_embeddings"] for v in embedding.values()]
        dense_embedding_matrix = torch.tensor(dense_embedding_list).squeeze(1)
        with open(pkl_save_path, 'wb') as f:
            pickle.dump(dense_embedding_matrix, f)

    get_dense_embedding_matrix(docs_embedding, '/home/ruzhi/work/mgte/data/docs_dense_embedding_matrix.pkl')
    get_dense_embedding_matrix(query_embedding, '/home/ruzhi/work/mgte/data/queries_dense_embedding_matrix.pkl')

    # #获取dense_embedding
    # docs_dense_embedding_list = [v["embedding"]["dense_embeddings"] for v in docs_embedding.values()]
    # # queries_dense_embedding_list = [v["query_embedding"]["dense_embeddings"] for v in query_embedding.values()]
    # queries_dense_embedding_list = [v["query_embedding"]["dense_embeddings"] for v in query_embedding.values()]

    # #将dense_embedding转化为tensor， batch_docs: [200000, 768]   batch_queries: [800, 768]
    # queries_dense_embedding_matrix = torch.tensor(queries_dense_embedding_list).squeeze(1)
    # with open('/home/ruzhi/work/mgte/data/queries_dense_embedding_matrix.pkl', 'wb') as f1:
    #     pickle.dump(queries_dense_embedding_matrix, f1)
    
    # docs_dense_embedding_matrix = torch.tensor(docs_dense_embedding_list).squeeze(1)
    # with open('/home/ruzhi/work/mgte/data/docs_dense_embedding_matrix.pkl', 'wb') as f2:
    #     pickle.dump(docs_dense_embedding_matrix, f2)

def get_one_hot_matrix(token_weights_list):
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-multilingual-base')
    vocab_size = len(tokenizer)
    print("词表大小为：", vocab_size)
    row, col, data = [], [], []
    with open("/home/ruzhi/work/mgte/data/vocab_dict.json",'r',encoding='utf-8') as file:
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

def prepare_token_weights_matrix(query_embedding, docs_embedding):

    def get_token_weights_matrix(embedding, pkl_save_path):
        # queries_token_weights_list = [v["query_embedding"]["token_weights"][0] for v in query_embedding.values()]
        token_weights_list = [v["embedding"]["token_weights"][0] for v in embedding.values()]
        token_weights_matrix = get_one_hot_matrix(token_weights_list)
        
        with open(pkl_save_path, 'wb') as f:
            pickle.dump(token_weights_matrix, f)

    
    # # queries_token_weights_list = [v["query_embedding"]["token_weights"][0] for v in query_embedding.values()]
    # queries_token_weights_list = [v["embedding"]["token_weights"][0] for v in query_embedding.values()]
    # queries_token_weights_matrix = get_one_hot_matrix(queries_token_weights_list)
    # print("得到query matrix")
    
    # with open('/home/ruzhi/work/mgte/data/new_query_token_weights_matrix.pkl', 'wb') as f1:
    #     pickle.dump(queries_token_weights_matrix, f1)
    get_token_weights_matrix(query_embedding, '/home/ruzhi/work/mgte/data/new_query_token_weights_matrix.pkl')
   
    # docs_token_weights_list = [v["embedding"]["token_weights"][0] for v in docs_embedding.values()]
    # docs_token_weights_matrix = get_one_hot_matrix(docs_token_weights_list)
    # print("得到docs matrix")
    
    # with open('/home/ruzhi/work/mgte/data/new_docs_token_weights_matrix.pkl', 'wb') as f2:
    #     pickle.dump(docs_token_weights_matrix, f2)
    
    get_token_weights_matrix(docs_embedding, '/home/ruzhi/work/mgte/data/new_docs_token_weights_matrix.pkl')

def prepare_embedding(doc_file_path, query_file_path):

    def get_embedding(model, json_read_path, json_save_path):
        dic = get_docs_data(json_read_path)
        for v in tqdm(dic.values()):
            v["embedding"] = model.encode(v["text"], return_dense = True, return_sparse = True)
        save_to_json(json_save_path, dic)
   
    model = GTEEmbedding('Alibaba-NLP/gte-multilingual-base')
    doc_file_path = "/home/ruzhi/work/mgte/data/corpus.jsonl"
    query_file_path = "/home/ruzhi/work/mgte/data/MLDR_test.jsonl"

    get_embedding(model, doc_file_path, json_save_path='/home/ruzhi/work/mgte/data/docs_embedding.json')
    get_embedding(model, query_file_path, json_save_path='/home/ruzhi/work/mgte/data/query_embedding.json')


# if __name__ == '__main__':
#     #读取带有embedding的数据
#     query_embedding = json_to_dict(FILE_PATH_PREFIX + "new_query_embedding.json")
#     docs_embedding = json_to_dict("/home/ruzhi/work/mgte/data/docs_embedding.json")
#     print("读取embedding数据完成!")
#     prepare_dense_embedding_matrix(query_embedding, docs_embedding)
#     prepare_token_weights_matrix(query_embedding, docs_embedding)
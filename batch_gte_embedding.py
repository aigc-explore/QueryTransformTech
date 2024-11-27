from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.utils import is_torch_npu_available

from utils import json_to_dict, read_pkl
import pickle
from GTEEmbedding import GTEEmbedding




def cal_cDCG(relevance_scores, k, len_doc_gt):
    def dcg_at_k(relevance_scores, k):
        relevance_scores = np.array(relevance_scores)[:k]
        dcg = np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))
        return dcg

    def idcg_at_k(k):
        i_relevance_scores = [0] * k
        for i in range(0, len_doc_gt):
            i_relevance_scores[i] = 1
        
        return dcg_at_k(i_relevance_scores, k)

    # 计算 cDCG@k
    def cdcg_at_k(relevance_scores, k):
        dcg = dcg_at_k(relevance_scores, k)
        idcg = idcg_at_k(k)
        return dcg / idcg if idcg > 0 else 0

    return cdcg_at_k(relevance_scores, k)

def compute_scores(top_k,
                   query_token_weights_path,
                   query_dense_embedding_path,
                   docs_token_weights_path,
                   docs_dense_embedding_path, 
                    dense_weight=1.0, 
                    sparse_weight=0.1, 
                    DENSE = True, 
                    SPARSE = True,
                    
                    ):
    if SPARSE:
        queries_token_weights_matrix = read_pkl(query_token_weights_path)
        docs_token_weights_matrix = read_pkl(docs_token_weights_path)
        sparse_result = torch.tensor(queries_token_weights_matrix.dot(docs_token_weights_matrix.T).toarray())
    else:
        sparse_result = None
    if DENSE:
        queries_dense_embedding_matrix = read_pkl(query_dense_embedding_path)
        docs_dense_embedding_matrix = read_pkl(docs_dense_embedding_path)
        dense_result = torch.mm(queries_dense_embedding_matrix, docs_dense_embedding_matrix.T, out=None)
    else:
        dense_result = None

    if sparse_result is not None and dense_result is not None:
        scores_matrix = dense_result * dense_weight + sparse_result * sparse_weight
    elif sparse_result is not None:
        scores_matrix = sparse_result
    elif dense_result is not None:
        scores_matrix = dense_result

    #获取每一行最大的前十个值及其索引
    top_k_elements, top_k_indices = torch.topk(scores_matrix, k=top_k, dim=1)

    return top_k_elements, top_k_indices


if __name__ == '__main__':
    query_token_weights_path = "/home/ruzhi/work/mgte/data/rewrite_query_token_weights_matrix.pkl"
    query_dense_embedding_path = "/home/ruzhi/work/mgte/data/rewrite_query_dense_embedding_matrix.pkl"
    docs_token_weights_path = "/home/ruzhi/work/mgte/data/new_docs_token_weights_matrix.pkl"
    docs_dense_embedding_path = "/home/ruzhi/work/mgte/data/docs_dense_embedding_matrix.pkl"

    #创建model
    model = GTEEmbedding('Alibaba-NLP/gte-multilingual-base')
    
    #读取带有embedding的数据
    query_embedding = json_to_dict("/home/ruzhi/work/mgte/data/rewrite_query_embedding.json")
    docs_embedding = json_to_dict("/home/ruzhi/work/mgte/data/docs_embedding.json")

    #获取所有docs的docid
    key_list = list(docs_embedding.keys())
    print("数据加载完成！")

    #获取每一行最大的前十个值及其索引
    k = 10
    total_cdcg = 0
    top_k_docs = []

    top_k_elements, top_k_indices = compute_scores(top_k=k, 
                                                   query_token_weights_path=query_token_weights_path, 
                                                   query_dense_embedding_path=query_dense_embedding_path,
                                                   docs_token_weights_path=docs_token_weights_path,
                                                   docs_dense_embedding_path=docs_dense_embedding_path)

    for index, value in enumerate(query_embedding.values()):
        docs_gt_id = [doc_gt['docid'] for doc_gt in value["doc"]]

        #根据索引获取最相关的10个docid
        relevant_k_doc_ids = [key_list[i] for i in top_k_indices[index]]

        # 保留最相关的前k个文档
        # relevant_k_docs = [docs_embedding[docid]["text"] for docid in relevant_k_doc_ids]
        # top_k_docs.append(relevant_k_docs)

        #将预测结果转为[1,0,0,1,0...]的格式
        top_k_prediction = [0] * k
        for i in range(0, k):
            if relevant_k_doc_ids[i] in docs_gt_id:
                top_k_prediction[i] = 1
        
        cdcg_at_k = cal_cDCG(top_k_prediction, k, len(docs_gt_id))
        total_cdcg += cdcg_at_k

    print("avg_cdcg:", total_cdcg / len(query_embedding))

# with open("top_20_chunks.pkl", "wb") as file:
#     pickle.dump(top_k_docs, file)

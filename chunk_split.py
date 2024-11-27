import tiktoken
import json
import pickle

from utils import get_docs_data,save_to_json
from GTEEmbedding import GTEEmbedding
from tqdm.notebook import tqdm




def split_text_by_tokens_tiktoken(encoder, text, token_limit=1024):
    """
    使用 tiktoken 按 token 数分块，若最后剩余 token 数不足 token_limit,则直接保存
    :param text: 输入文本
    :param token_limit: 每块的最大 token 数
    :return: 分块后的字符串列表
    """

    tokens = encoder.encode(text)
    chunks = []

    # 循环分块
    for i in range(0, len(tokens), token_limit):
        # 如果剩余的 token 数小于 token_limit，直接保存剩余部分
        if i + token_limit >= len(tokens):
            chunk_tokens = tokens[i:]  # 剩余 token
        else:
            chunk_tokens = tokens[i:i + token_limit]  # 当前块的 token
        
        # 将 token 解码回文本
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

        # 如果已经保存了最后一部分，退出循环
        if i + token_limit >= len(tokens):
            break
    return chunks


if __name__ == '__main__':
    model = GTEEmbedding('Alibaba-NLP/gte-multilingual-base')
    encoder = tiktoken.get_encoding("cl100k_base")
    doc_file_path = "/home/ruzhi/work/mgte/data/MLDR/en/corpus.jsonl"
    split_docs_embedding_pkl_path = "/home/ruzhi/work/mgte/data/split_chunk_1k_docs_embedding.pkl"
    split_docs_json_save_path = "/home/ruzhi/work/mgte/data/split_chunk_corpus_1k.json"
    docs_dic = get_docs_data(doc_file_path)
    res_dic = {}
    count = 0
    for k,v in docs_dic.items():
        doc_chunks = split_text_by_tokens_tiktoken(encoder, v["text"])
        for chunk in doc_chunks:
            chunk_id = str(count)
            res_dic[chunk_id] = {"doc_id": k, "text": chunk}
            count += 1
    # 储存新的json文件
    with open("/home/ruzhi/work/mgte/data/split_chunk_corpus_1k.json","w",encoding='utf-8') as file:
        json.dump(res_dic, file, indent=4, ensure_ascii=False)


    # for v in tqdm(res_dic.values()):
    #     v["embedding"] = model.encode(v["text"], return_dense = True, return_sparse = True)
    
    #太大了，写不进json了
    #save_to_json("/home/ruzhi/work/mgte/data/split_chunk_1k_docs_embedding.json", res_dic)
   
    with open(split_docs_embedding_pkl_path, "wb") as file:
        pickle.dump(res_dic, file)
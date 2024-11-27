import json
import pickle


def save_to_pkl(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
def read_pkl(file_path):
    with open(file_path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data

def get_docs_data(file_path):
    docs_dic = {}
    with open(file_path,'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            docs_dic[data['docid']] = {
                                        "text": data['text'],
                                        "embedding": []
                                        }
    return docs_dic

def get_query_data(file_path):
    query_dic = {}
    with open(file_path,'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            query_dic[data['query_id']] = {
                                            "query": data["query"],
                                            "doc": data['positive_passages']     #字典类型 [{'docid': 'doc-zh-<...>', 'text': '...'}, ...]
                                        }
    return query_dic

def save_to_json(file_path, dic):
    # 将字典转换为 JSON 格式并保存到文件
    with open(file_path, 'w') as json_file:
        json.dump(dic, json_file, indent=4)

def json_to_dict(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


import json
import random
from typing import List, Dict, Tuple
from openai import OpenAI
import re
import json
from tqdm import tqdm

def clean_llm_output(text: str) -> str:
    """
    Remove <think>...</think> blocks if they exist.
    """
    # 删除 <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 去掉前后空白
    cleaned = cleaned.strip()

    return cleaned

def extract_selected_index(llm_output: str):
    # 1️⃣ 先清理 think 部分
    cleaned_output = clean_llm_output(llm_output)

    # 2️⃣ 尝试解析 JSON
    try:
        data = json.loads(cleaned_output)
        return int(data["selected_index"])
    except:
        # 3️⃣ 如果 JSON 失败，提取第一个数字
        match = re.search(r'\d+', cleaned_output)
        if match:
            return int(match.group())
        return None

############################################
# 1️⃣ 读取 JSON 文件
############################################

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


############################################
# 2️⃣ 构建 ID → label 映射
############################################

def build_entity_dict(entity_list: List[Dict]) -> Dict[int, Dict]:
    entity_dict = {}
    for item in entity_list:
        entity_dict[item["value"]] = {
            "label": item["label"],
            "classname": item["classname"]
        }
    return entity_dict


def build_relation_dict(relation_list: List[Dict]) -> Dict[int, str]:
    relation_dict = {}
    for item in relation_list:
        relation_dict[int(item["id"])] = item["label"]
    return relation_dict


############################################
# 3️⃣ 读取三元组
############################################

def load_triples(path: str) -> List[Tuple[int, int, int]]:
    triples = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过第一行数量
            h, t, r = line.strip().split()
            triples.append((int(h), int(t), int(r)))
    return triples


############################################
# 4️⃣ 抽取随机样本
############################################

def sample_triples(triples: List[Tuple[int, int, int]], n: int = 10):
    return random.sample(triples, n)


############################################
# 🆕 生成候选实体（包含正确答案）
############################################

def build_tail_candidates(true_tail_id: int,
                          entity_dict: Dict,
                          num_candidates: int = 10) -> List[str]:
    
    all_entity_ids = list(entity_dict.keys())
    
    # 移除真实 tail
    all_entity_ids.remove(true_tail_id)
    
    # 随机采样 num_candidates - 1 个负样本
    negative_ids = random.sample(all_entity_ids, num_candidates - 1)
    
    # 加入真实 tail
    candidate_ids = negative_ids + [true_tail_id]
    
    # 打乱顺序
    random.shuffle(candidate_ids)
    
    # 转换为 label
    candidates = [entity_dict[eid]["label"] for eid in candidate_ids]
    
    return candidates
############################################
# 🔁 修改：Constrained Prompt
############################################

def build_prompt(head: str,
                 relation: str,
                 candidates: List[str]) -> str:
    
    indexed_candidates = "\n".join(
        [f"{i}. {c}" for i, c in enumerate(candidates)]
    )

    prompt = f"""
You are an expert knowledge graph completion system.

Your task is to select the correct tail entity from a candidate list.

Triple format:
(head entity, relation, tail entity)

Important:
- You are predicting ONLY the tail entity.
- The head entity "{head}" is NOT a valid answer.
- You MUST select exactly ONE candidate index.
- You MUST NOT output entity names.
- You MUST NOT generate new entities.
- Output ONLY valid JSON.

Before answering, internally:
1. Understand the semantic meaning of the relation.
2. Determine the expected type of the tail entity.
3. Compare all candidates carefully.
4. Select the most logically consistent one.
Do NOT output reasoning.

Candidate entities:
{indexed_candidates}

Output format:

{{
  "selected_index": integer
}}


Incomplete triple:
({head}, {relation}, ?)
"""
    return prompt


############################################
# 6️⃣ 调用 OpenAI API
############################################

def query_llm(prompt: str) -> str:
    client = OpenAI(
        base_url="http://localhost:22014/v1",
        api_key="EMPTY"   # vLLM 允许随便填
    )

    resp = client.chat.completions.create(
        model="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": prompt}]
    )

    # print(resp.choices[0].message.content)
    return resp.choices[0].message.content


############################################
# 7️⃣ 单个三元组预测
############################################

def predict_for_triple(triple: Tuple[int, int, int],
                       entity_dict: Dict,
                       relation_dict: Dict):
    
    h_id, t_id, r_id = triple
    
    head = entity_dict[h_id]["label"]
    relation = relation_dict[r_id]
    true_tail = entity_dict[t_id]["label"]
    
    # 🆕 构建候选集合
    candidates = build_tail_candidates(t_id, entity_dict, num_candidates=10)
    
    prompt = build_prompt(head, relation, candidates)
    # print(prompt)
    
    result = query_llm(prompt)
    
    return {
        "triple": (head, relation, true_tail),
        "candidates": candidates,
        "llm_prediction": result
    }

def compute_hit_at_1(results: List[Dict]) -> float:
    hit = 0
    total = len(results)

    for item in results:
        true_tail = item["triple"][2]
        candidates = item["candidates"]
        llm_output = item["llm_prediction"]

        pred_index = extract_selected_index(llm_output)

        # 输出结果
        print("Original Triple:", item["triple"])
        print("candidates:", item["candidates"])
        print("LLM Prediction:", pred_index)
        print("=" * 50)

        if pred_index is None:
            continue

        # 防止 index 越界
        if 0 <= pred_index < len(candidates):
            predicted_tail = candidates[pred_index]

            if predicted_tail == true_tail:
                hit += 1

    if total == 0:
        return 0.0

    return hit / total

############################################
# 8️⃣ 主流程
############################################

def main():
    
    entity_path = "/home/wenbin.guo/RAG/data/FB15k-237/entity.json"
    relation_path = "/home/wenbin.guo/RAG/data/FB15k-237/relation.json"
    triple_path = "/home/wenbin.guo/RAG/data/FB15k-237/train2id.txt"  # 修改为你的三元组文件
    
    # 加载数据
    entity_list = load_json(entity_path)
    relation_list = load_json(relation_path)
    triples = load_triples(triple_path)
    
    entity_dict = build_entity_dict(entity_list)
    relation_dict = build_relation_dict(relation_list)
    
    # 抽取10个样本
    sampled_triples = sample_triples(triples, 2000)
    
    results = []
    
    for triple in tqdm(sampled_triples, desc="Evaluating", ncols=100):
        prediction = predict_for_triple(triple, entity_dict, relation_dict)
        results.append(prediction)
    
    # 输出结果
    # for item in results:
    #     print("Original Triple:", item["triple"])
    #     print("candidates:", item["candidates"])
    #     print("LLM Prediction:", item["llm_prediction"])
    #     print("=" * 50)
    
    hit1 = compute_hit_at_1(results)
    print(f"\nHit@1: {hit1:.4f}")


if __name__ == "__main__":
    main()
    
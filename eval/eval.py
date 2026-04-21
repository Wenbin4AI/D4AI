import os
import json
import re
import ast
import time
import random
import argparse
from typing import List, Dict, Any
from openai import OpenAI

# =========================
# LLM Client
# =========================
def build_client(base_url: str, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)

# =========================
# 文件读写
# =========================
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# =========================
# JSON 修复 / 检查
# =========================
def fix_json_gold(meta: Dict[str, Any]):
    """确保每个 JSON 只有一个 gold"""
    gold_tail = meta["query"]["gold_tail"]
    for cand in meta["query"]["selected_candidates"]:
        cand_id = cand.get("candidate_id")
        cand["is_gold"] = 1 if cand_id == gold_tail else 0
    meta["candidate_limit"] = len(meta["query"]["selected_candidates"])
    return meta

def infer_candidate_count(meta: Dict[str, Any]) -> int:
    if "query" in meta and "selected_candidates" in meta["query"]:
        return len(meta["query"]["selected_candidates"])
    if "selected_candidates" in meta:
        return len(meta["selected_candidates"])
    if "candidate_limit" in meta:
        return int(meta["candidate_limit"])
    raise ValueError("无法解析候选数量")

def extract_gold_rank(meta: Dict[str, Any]) -> int:
    return int(meta["query"]["gold_candidate_rank_index"])

# =========================
# LLM 调用与解析
# =========================
def query_model(client: OpenAI, model_name: str, prompt: str, temperature=0.0) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

def parse_rank_list(text: str, expected_n: int) -> List[int]:
    text = re.sub(r"<think.*?>", "", text, flags=re.S | re.I)
    text = re.sub(r"</think>", "", text, flags=re.S | re.I).strip()
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return [int(x) for x in obj]
    except:
        pass
    matches = re.findall(r"\[[\s\d,]+\]", text)
    for candidate in reversed(matches):
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, list):
                return [int(x) for x in obj]
        except:
            continue
    # fallback: 返回顺序 1..N
    return list(range(1, expected_n + 1))

# =========================
# 单样本 Hit@1 测试
# =========================
def evaluate_one(client, model_name: str, txt_path: str, json_path: str, temperature=0.0) -> Dict[str, Any]:
    prompt = load_text(txt_path)
    meta = load_json(json_path)
    meta = fix_json_gold(meta)

    candidate_count = infer_candidate_count(meta)
    gold_rank = extract_gold_rank(meta)

    start = time.time()
    raw_output = query_model(client, model_name, prompt, temperature)
    latency = time.time() - start

    final_rank = parse_rank_list(raw_output, candidate_count)
    rank_of_gold = final_rank.index(gold_rank) + 1
    hit1 = 1.0 if rank_of_gold == 1 else 0.0
    mrr = 1.0 / rank_of_gold

    return {
        "txt_file": os.path.basename(txt_path),
        "json_file": os.path.basename(json_path),
        "rank_of_gold": rank_of_gold,
        "hit@1": hit1,
        "mrr": mrr,
        "latency_sec": latency,
        "model_output_raw": raw_output,
        "model_output_final": final_rank
    }

# =========================
# 批量收集 prompt/json
# =========================
def collect_prompt_pairs(prompt_dir: str) -> List[Dict[str, str]]:
    pairs = []
    for name in os.listdir(prompt_dir):
        if not name.endswith(".txt"):
            continue
        txt_path = os.path.join(prompt_dir, name)
        json_path = os.path.join(prompt_dir, os.path.splitext(name)[0] + ".json")
        if os.path.exists(json_path):
            pairs.append({"txt_path": txt_path, "json_path": json_path})
    return sorted(pairs, key=lambda x: x["txt_path"])

# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", type=str, default="/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_hit1")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B")
    parser.add_argument("--base_url", type=str, default="http://localhost:22014/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_json", type=str, default="eval_hit1.json")
    args = parser.parse_args()

    random.seed(args.seed)
    client = build_client(args.base_url, args.api_key)
    pairs = collect_prompt_pairs(args.prompt_dir)

    sampled_pairs = random.sample(pairs, min(args.sample_size, len(pairs)))

    results = []
    for i, pair in enumerate(sampled_pairs, 1):
        txt_path, json_path = pair["txt_path"], pair["json_path"]
        try:
            res = evaluate_one(client, args.model_name, txt_path, json_path, args.temperature)
            results.append(res)
            print(f"[{i}/{len(sampled_pairs)}] {os.path.basename(txt_path)} Hit@1={res['hit@1']} rank_of_gold={res['rank_of_gold']} latency={res['latency_sec']:.2f}s")
        except Exception as e:
            print(f"[{i}/{len(sampled_pairs)}] {os.path.basename(txt_path)} Failed: {e}")

    hit1_avg = sum(x["hit@1"] for x in results) / len(results) if results else 0.0
    avg_latency = sum(x["latency_sec"] for x in results) / len(results) if results else 0.0

    print(f"\n==== Hit@1 Summary ====")
    print(f"Samples evaluated: {len(results)}")
    print(f"Hit@1: {hit1_avg:.4f}")
    print(f"Average latency: {avg_latency:.2f}s")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "hit@1_avg": hit1_avg, "avg_latency_sec": avg_latency}, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
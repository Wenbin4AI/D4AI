import os
import re
import json
from glob import glob
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm


def clean_llm_output(text: str) -> str:
    """
    Remove <think>...</think> blocks if they exist.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def extract_selected_index(llm_output: str) -> Optional[int]:
    """
    从模型输出中提取 selected_index
    优先解析 JSON；失败则回退提取第一个数字
    """
    cleaned_output = clean_llm_output(llm_output)

    try:
        data = json.loads(cleaned_output)
        return int(data["selected_index"])
    except Exception:
        pass

    match = re.search(r"\d+", cleaned_output)
    if match:
        return int(match.group())

    return None


def load_json_safe(path: str) -> Optional[Dict]:
    """
    安全读取 JSON：
    - 空文件 -> 返回 None
    - 非法 JSON -> 返回 None
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print(f"[Skip Empty JSON] {path}")
            return None

        return json.loads(text)

    except Exception as e:
        print(f"[Skip Bad JSON] {path} | {e}")
        return None


def extract_file_index(filename: str) -> int:
    """
    从 test_20463.json 中提取数字 20463，用于排序
    """
    base = os.path.basename(filename)
    match = re.search(r"(\d+)", base)
    return int(match.group(1)) if match else 10**18


def load_prompt_cases(prompt_dir: str) -> List[Dict]:
    """
    遍历目录下所有 json 文件，并按文件编号排序
    遇到损坏/空文件直接跳过
    """
    paths = glob(os.path.join(prompt_dir, "*.json"))
    paths = sorted(paths, key=extract_file_index)

    cases = []
    skipped = 0

    for path in paths:
        item = load_json_safe(path)
        if item is None:
            skipped += 1
            continue

        required_keys = [
            "head_label",
            "relation_label",
            "gold_selected_index",
            "candidate_lines"
        ]
        missing = [k for k in required_keys if k not in item]
        if missing:
            print(f"[Skip Invalid Case] {path} | missing keys: {missing}")
            skipped += 1
            continue

        item["_file_path"] = path
        item["_file_name"] = os.path.basename(path)
        cases.append(item)

    print(f"Loaded valid cases: {len(cases)}")
    print(f"Skipped files: {skipped}")
    return cases


def format_query_evidence(query_evidence: List[str]) -> str:
    if not query_evidence:
        return ""

    evidence_lines = "\n".join([f"- {x}" for x in query_evidence])
    return f"\nKey evidence:\n{evidence_lines}\n"


def format_candidate_lines(candidate_lines: List[Dict]) -> str:
    lines = []
    for item in candidate_lines:
        if "text" in item and item["text"]:
            lines.append(item["text"])
        else:
            idx = item["selected_index"]
            label = item["label"]
            classname = item.get("classname", "Unknown")
            lines.append(f"{idx}. {label} [class={classname}]")
    return "\n".join(lines)


def build_prompt_from_case(case: Dict) -> str:
    head = case["head_label"]
    relation = case["relation_label"]
    candidate_lines = case["candidate_lines"]
    query_evidence = case.get("query_evidence", [])

    indexed_candidates = format_candidate_lines(candidate_lines)
    evidence_block = format_query_evidence(query_evidence)

    prompt = f"""You are an expert knowledge graph completion system.

Your task is to select the correct tail entity from a candidate list.

Triple format:
(head entity, relation, tail entity)

Important:
- You are predicting ONLY the tail entity.
- The head entity "{head}" is NOT a valid answer.
- You MUST select exactly ONE candidate index.
- You MUST NOT output entity names.
- You MUST NOT generate new entities.
- Use the ontology and structural evidence to compare candidates.
- Output ONLY valid JSON.

Before answering, internally:
1. Understand the semantic meaning of the relation.
2. Determine the expected type of the tail entity.
3. Compare all candidates carefully.
4. Select the most logically consistent candidate.
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


def query_llm(prompt: str) -> str:
    client = OpenAI(
        base_url="http://localhost:22014/v1",
        api_key="EMPTY"
    )

    resp = client.chat.completions.create(
        model="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content


def predict_for_case(case: Dict) -> Dict:
    prompt = build_prompt_from_case(case)
    # print(prompt)
    result = query_llm(prompt)
    pred_index = extract_selected_index(result)

    gold_index = case["gold_selected_index"]
    candidate_lines = case["candidate_lines"]

    predicted_tail = None
    true_tail = None
    is_hit = 0

    for c in candidate_lines:
        if c["selected_index"] == gold_index:
            true_tail = c["label"]
            break

    if pred_index is not None:
        for c in candidate_lines:
            if c["selected_index"] == pred_index:
                predicted_tail = c["label"]
                break

    if pred_index == gold_index:
        is_hit = 1

    return {
        "file": case.get("_file_name"),
        "triple": (
            case["head_label"],
            case["relation_label"],
            true_tail
        ),
        "gold_index": gold_index,
        "pred_index": pred_index,
        "predicted_tail": predicted_tail,
        "true_tail": true_tail,
        "llm_prediction": result,
        "is_hit": is_hit,
        "prompt": prompt,
    }


def main():
    prompt_dir = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v5"

    cases = load_prompt_cases(prompt_dir)
    print(f"==== Start evaluating {len(cases)} valid prompt files ====")

    results = []
    hit_count = 0
    valid_count = 0

    for idx, case in enumerate(tqdm(cases, desc="Evaluating", ncols=100), start=1):
        try:
            prediction = predict_for_case(case)
            results.append(prediction)

            hit_count += prediction["is_hit"]
            valid_count += 1

            print(
                f"[{idx}/{len(cases)}] {prediction['file']} | "
                f"pred={prediction['pred_index']} | "
                f"gold={prediction['gold_index']} | "
                f"hit1={prediction['is_hit']}"
            )

            if idx % 50 == 0:
                current_hit1 = hit_count / valid_count if valid_count > 0 else 0.0
                print(f"\n===== Processed {idx} samples =====")
                print(f"Current Hit@1: {current_hit1:.4f}")
                print("Last Triple:", prediction["triple"])
                print("Pred Index:", prediction["pred_index"])
                print("Gold Index:", prediction["gold_index"])
                print("Pred Tail:", prediction["predicted_tail"])
                print("True Tail:", prediction["true_tail"])
                print("Hit:", prediction["is_hit"])
                print("=" * 60)

        except Exception as e:
            print(f"[{idx}/{len(cases)}] {case.get('_file_name', 'unknown')} | Failed | {e}")

    final_hit1 = hit_count / valid_count if valid_count > 0 else 0.0
    print(f"\nFinal Hit@1: {final_hit1:.4f}")
    print(f"Valid cases: {valid_count}/{len(cases)}")


if __name__ == "__main__":
    main()
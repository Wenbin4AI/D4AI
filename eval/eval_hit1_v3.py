import os
import re
import ast
import json
import time
from typing import Optional, List, Tuple
from openai import OpenAI


# =========================================================
# 1. LLM 客户端配置
# =========================================================

client = OpenAI(
    base_url="http://localhost:22014/v1",
    api_key="EMPTY"
)

MODEL_PATH = "/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B"


# =========================================================
# 2. 解析模型输出
#    目标：从输出中尽量稳健地提取一个候选编号排序列表
# =========================================================

def extract_first_list(text: str) -> Optional[List[int]]:
    """
    从模型输出中提取第一个 Python 风格列表，例如:
    [3, 7, 1, 5, 2, ...]
    """
    if not text:
        return None

    # 去掉 <think> ... </think> 干扰
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 找第一个 [...] 结构
    match = re.search(r"\[[\s\d,]+\]", text)
    if not match:
        return None

    candidate = match.group(0)
    try:
        result = ast.literal_eval(candidate)
        if isinstance(result, list) and all(isinstance(x, int) for x in result):
            return result
    except Exception:
        return None

    return None


def extract_one_number(text: str) -> Optional[int]:
    """
    如果模型没有输出列表，只输出一个数字，就取第一个数字。
    """
    if not text:
        return None

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\b\d+\b", text)
    if match:
        return int(match.group(0))
    return None


def parse_prediction(text: str, candidate_count: int) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    解析模型输出，优先解析完整排序列表。
    返回:
      - parsed_ranking: 解析出的编号列表
      - error_msg: 若失败则返回错误信息
    """
    ranking = extract_first_list(text)
    if ranking is not None:
        # 基本合法性检查
        expected = set(range(1, candidate_count + 1))
        pred_set = set(ranking)

        if len(ranking) != candidate_count:
            return None, f"排序列表长度不正确，期望 {candidate_count}，实际 {len(ranking)}"

        if pred_set != expected:
            return None, (
                f"排序列表编号不合法，期望为 1~{candidate_count} 且不重复，"
                f"实际得到: {ranking}"
            )

        return ranking, None

    # 兜底：如果只输出一个数，就补成“第1预测”
    one_num = extract_one_number(text)
    if one_num is not None:
        if 1 <= one_num <= candidate_count:
            remaining = [x for x in range(1, candidate_count + 1) if x != one_num]
            return [one_num] + remaining, None
        return None, f"模型输出了数字 {one_num}，但不在合法范围 1~{candidate_count}"

    return None, "无法解析模型输出为候选编号列表"


# =========================================================
# 3. 调用 LLM
# =========================================================

def call_llm(prompt: str,
             model_path: str = MODEL_PATH,
             temperature: float = 0.0,
             max_tokens: int = 512,
             timeout: int = 300) -> str:
    """
    调用本地 vLLM OpenAI 接口，返回原始文本输出
    """
    resp = client.chat.completions.create(
        model=model_path,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return resp.choices[0].message.content


# =========================================================
# 4. 评测单个 prompt 文件
# =========================================================

def evaluate_one_file(prompt_json_path: str,
                      model_path: str = MODEL_PATH,
                      temperature: float = 0.0,
                      max_tokens: int = 512,
                      timeout: int = 300,
                      verbose: bool = False) -> dict:
    """
    读取单个 *_prompt.json 文件并评测
    """
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data["prompt"]
    gold_rank_id = data.get("gold_rank_id_in_prompt", None)
    candidate_lines = data.get("candidate_lines", [])
    file_name = data.get("file", os.path.basename(prompt_json_path))

    if gold_rank_id is None:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": "gold_rank_id_in_prompt 缺失，无法评测",
            "raw_output": None,
            "predicted_top1": None,
            "gold_rank_id_in_prompt": None,
        }

    candidate_count = len(candidate_lines)
    if candidate_count == 0:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": "candidate_lines 为空，无法评测",
            "raw_output": None,
            "predicted_top1": None,
            "gold_rank_id_in_prompt": gold_rank_id,
        }

    start_time = time.time()

    try:
        raw_output = call_llm(
            prompt=prompt,
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except Exception as e:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": f"LLM 调用失败: {e}",
            "raw_output": None,
            "predicted_top1": None,
            "gold_rank_id_in_prompt": gold_rank_id,
            "elapsed_sec": round(time.time() - start_time, 4),
        }

    ranking, parse_error = parse_prediction(raw_output, candidate_count)

    if ranking is None:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": parse_error,
            "raw_output": raw_output,
            "predicted_top1": None,
            "gold_rank_id_in_prompt": gold_rank_id,
            "elapsed_sec": round(time.time() - start_time, 4),
        }

    pred_top1 = ranking[0]
    hit1 = int(pred_top1 == gold_rank_id)

    if verbose:
        print(f"[DEBUG] file={file_name}")
        print(f"[DEBUG] gold={gold_rank_id}, pred_top1={pred_top1}, hit1={hit1}")
        print(f"[DEBUG] raw_output={raw_output}")

    return {
        "file": file_name,
        "success": True,
        "hit1": hit1,
        "error": None,
        "raw_output": raw_output,
        "parsed_ranking": ranking,
        "predicted_top1": pred_top1,
        "gold_rank_id_in_prompt": gold_rank_id,
        "elapsed_sec": round(time.time() - start_time, 4),
    }


# =========================================================
# 5. 批量评测整个目录
# =========================================================

def evaluate_prompt_directory(prompt_dir: str,
                              save_result_path: Optional[str] = None,
                              model_path: str = MODEL_PATH,
                              temperature: float = 0.0,
                              max_tokens: int = 512,
                              timeout: int = 300,
                              show_every: int = 50,
                              limit: Optional[int] = None,
                              verbose: bool = False) -> dict:
    """
    批量评测目录下所有 *_prompt.json 文件
    """
    files = sorted([
        f for f in os.listdir(prompt_dir)
        if f.endswith(".json")
    ])

    if limit is not None:
        files = files[:limit]

    total = len(files)
    if total == 0:
        raise ValueError(f"目录下没有找到 json 文件: {prompt_dir}")

    results = []
    success_count = 0
    hit_count = 0
    parse_fail_count = 0
    call_fail_count = 0

    print(f"==== Start evaluating {total} prompt files ====")

    for idx, fname in enumerate(files, start=1):
        path = os.path.join(prompt_dir, fname)
        result = evaluate_one_file(
            prompt_json_path=path,
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            verbose=verbose
        )
        results.append(result)

        if result["success"]:
            success_count += 1
            hit_count += result["hit1"]
        else:
            err = result.get("error", "") or ""
            if "LLM 调用失败" in err:
                call_fail_count += 1
            else:
                parse_fail_count += 1

        if result["success"]:
            print(
                f"[{idx}/{total}] {fname} | "
                f"pred_top1={result['predicted_top1']} | "
                f"gold={result['gold_rank_id_in_prompt']} | "
                f"hit1={result['hit1']}"
            )
        else:
            print(f"[{idx}/{total}] {fname} | Failed | {result['error']}")

        if idx % show_every == 0 or idx == total:
            current_hit1_all = hit_count / idx
            current_hit1_valid = hit_count / success_count if success_count > 0 else 0.0
            print(
                f"---- Progress {idx}/{total} | "
                f"Hit@1(all)={current_hit1_all:.4f} | "
                f"Hit@1(valid_only)={current_hit1_valid:.4f} | "
                f"success={success_count} | "
                f"parse_fail={parse_fail_count} | call_fail={call_fail_count}"
            )

    final_hit1_all = hit_count / total
    final_hit1_valid = hit_count / success_count if success_count > 0 else 0.0

    summary = {
        "prompt_dir": prompt_dir,
        "model_path": model_path,
        "total_files": total,
        "success_count": success_count,
        "parse_fail_count": parse_fail_count,
        "call_fail_count": call_fail_count,
        "hit_count": hit_count,
        "hit1_all": round(final_hit1_all, 6),
        "hit1_valid_only": round(final_hit1_valid, 6),
        "results": results,
    }

    print("\n==== Final Result ====")
    print(f"Total files        : {total}")
    print(f"Success            : {success_count}")
    print(f"Parse fail         : {parse_fail_count}")
    print(f"Call fail          : {call_fail_count}")
    print(f"Hit count          : {hit_count}")
    print(f"Final Hit@1 (all)  : {final_hit1_all:.6f}")
    print(f"Final Hit@1 (valid): {final_hit1_valid:.6f}")

    if save_result_path is not None:
        os.makedirs(os.path.dirname(save_result_path), exist_ok=True)
        with open(save_result_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved evaluation result to: {save_result_path}")

    return summary


# =========================================================
# 6. main
# =========================================================

if __name__ == "__main__":
    prompt_dir = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v4"
    save_result_path = "/home/wenbin.guo/DKGE4R/eval/test_prompts_v3_eval_result.json"

    evaluate_prompt_directory(
        prompt_dir=prompt_dir,
        save_result_path=save_result_path,
        model_path=MODEL_PATH,
        temperature=0.0,
        max_tokens=512,
        timeout=300,
        show_every=50,
        limit=None,        # 先小规模测试可改成 20
        verbose=False
    )
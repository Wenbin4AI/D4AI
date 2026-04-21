import os
import re
import json
import time
from typing import Optional, Dict, Any, List
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
# 2. 输出清洗与解析
#    参考你效果较好的那版逻辑：先去掉 <think>，再解析 JSON，
#    JSON 失败时回退到提取第一个整数。:contentReference[oaicite:1]{index=1}
# =========================================================

def clean_llm_output(text: str) -> str:
    """
    Remove <think>...</think> blocks if they exist.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def extract_selected_index(llm_output: str) -> Optional[int]:
    """
    优先解析:
    {
      "selected_index": 3
    }

    如果 JSON 解析失败，则提取第一个整数作为兜底。
    """
    cleaned_output = clean_llm_output(llm_output)

    # 1) 直接尝试完整 JSON
    try:
        data = json.loads(cleaned_output)
        if isinstance(data, dict) and "selected_index" in data:
            return int(data["selected_index"])
    except Exception:
        pass

    # 2) 提取代码块中的 JSON
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_output, flags=re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "selected_index" in data:
                return int(data["selected_index"])
        except Exception:
            pass

    # 3) 宽松提取 selected_index: number
    key_match = re.search(r'"selected_index"\s*:\s*(-?\d+)', cleaned_output)
    if key_match:
        try:
            return int(key_match.group(1))
        except Exception:
            pass

    # 4) 兜底：提取第一个整数
    first_num = re.search(r'-?\d+', cleaned_output)
    if first_num:
        try:
            return int(first_num.group(0))
        except Exception:
            return None

    return None


# =========================================================
# 3. 调用模型
# =========================================================

def query_llm(
    prompt: str,
    model_path: str = MODEL_PATH,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 300
) -> str:
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

def evaluate_one_prompt_file(
    prompt_json_path: str,
    model_path: str = MODEL_PATH,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 300,
    verbose: bool = False
) -> Dict[str, Any]:
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = data.get("file", os.path.basename(prompt_json_path))
    prompt = data.get("prompt", "")
    gold_selected_index = data.get("gold_selected_index", None)
    candidate_lines = data.get("candidate_lines", [])

    if not prompt:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": "prompt 为空",
            "raw_output": None,
            "pred_selected_index": None,
            "gold_selected_index": gold_selected_index,
            "elapsed_sec": 0.0,
        }

    if gold_selected_index is None:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": "gold_selected_index 缺失，无法评测",
            "raw_output": None,
            "pred_selected_index": None,
            "gold_selected_index": None,
            "elapsed_sec": 0.0,
        }

    if not candidate_lines:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": "candidate_lines 为空，无法评测",
            "raw_output": None,
            "pred_selected_index": None,
            "gold_selected_index": gold_selected_index,
            "elapsed_sec": 0.0,
        }

    candidate_count = len(candidate_lines)
    start_time = time.time()

    try:
        raw_output = query_llm(
            prompt=prompt,
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    except Exception as e:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": f"LLM 调用失败: {e}",
            "raw_output": None,
            "pred_selected_index": None,
            "gold_selected_index": gold_selected_index,
            "elapsed_sec": round(time.time() - start_time, 4),
        }

    pred_selected_index = extract_selected_index(raw_output)

    if pred_selected_index is None:
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": "无法解析 selected_index",
            "raw_output": raw_output,
            "pred_selected_index": None,
            "gold_selected_index": gold_selected_index,
            "elapsed_sec": round(time.time() - start_time, 4),
        }

    if not (0 <= pred_selected_index < candidate_count):
        return {
            "file": file_name,
            "success": False,
            "hit1": 0,
            "error": f"selected_index 越界: {pred_selected_index}, 合法范围为 0~{candidate_count - 1}",
            "raw_output": raw_output,
            "pred_selected_index": pred_selected_index,
            "gold_selected_index": gold_selected_index,
            "elapsed_sec": round(time.time() - start_time, 4),
        }

    hit1 = int(pred_selected_index == gold_selected_index)

    pred_tail = None
    gold_tail = None
    try:
        pred_tail = candidate_lines[pred_selected_index]["label"]
    except Exception:
        pred_tail = None

    try:
        gold_tail = candidate_lines[gold_selected_index]["label"]
    except Exception:
        gold_tail = None

    if verbose:
        print(f"[DEBUG] file={file_name}")
        print(f"[DEBUG] pred_selected_index={pred_selected_index}, gold_selected_index={gold_selected_index}, hit1={hit1}")
        print(f"[DEBUG] pred_tail={pred_tail}, gold_tail={gold_tail}")
        print(f"[DEBUG] raw_output={raw_output}")

    return {
        "file": file_name,
        "success": True,
        "hit1": hit1,
        "error": None,
        "raw_output": raw_output,
        "cleaned_output": clean_llm_output(raw_output),
        "pred_selected_index": pred_selected_index,
        "gold_selected_index": gold_selected_index,
        "pred_tail": pred_tail,
        "gold_tail": gold_tail,
        "elapsed_sec": round(time.time() - start_time, 4),
    }


# =========================================================
# 5. 批量评测整个目录
# =========================================================

def evaluate_prompt_directory(
    prompt_dir: str,
    save_result_path: Optional[str] = None,
    model_path: str = MODEL_PATH,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 300,
    show_every: int = 50,
    limit: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    files = sorted([
        f for f in os.listdir(prompt_dir)
        if f.endswith(".json")
    ])

    if limit is not None:
        files = files[:limit]

    total = len(files)
    if total == 0:
        raise ValueError(f"目录下没有找到 json 文件: {prompt_dir}")

    print(f"==== Start evaluating {total} prompt files ====")

    results: List[Dict[str, Any]] = []
    success_count = 0
    hit_count = 0
    parse_fail_count = 0
    call_fail_count = 0
    other_fail_count = 0

    for idx, fname in enumerate(files, start=1):
        path = os.path.join(prompt_dir, fname)

        result = evaluate_one_prompt_file(
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
            print(
                f"[{idx}/{total}] {fname} | "
                f"pred={result['pred_selected_index']} | "
                f"gold={result['gold_selected_index']} | "
                f"hit1={result['hit1']}"
            )
        else:
            err = result.get("error", "") or ""
            if "LLM 调用失败" in err:
                call_fail_count += 1
            elif "无法解析 selected_index" in err or "越界" in err:
                parse_fail_count += 1
            else:
                other_fail_count += 1

            print(f"[{idx}/{total}] {fname} | Failed | {err}")

        if idx % show_every == 0 or idx == total:
            hit1_all = hit_count / idx
            hit1_valid = hit_count / success_count if success_count > 0 else 0.0
            print(
                f"---- Progress {idx}/{total} | "
                f"Hit@1(all)={hit1_all:.4f} | "
                f"Hit@1(valid_only)={hit1_valid:.4f} | "
                f"success={success_count} | "
                f"parse_fail={parse_fail_count} | "
                f"call_fail={call_fail_count} | "
                f"other_fail={other_fail_count}"
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
        "other_fail_count": other_fail_count,
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
    print(f"Other fail         : {other_fail_count}")
    print(f"Hit count          : {hit_count}")
    print(f"Final Hit@1 (all)  : {final_hit1_all:.6f}")
    print(f"Final Hit@1 (valid): {final_hit1_valid:.6f}")

    if save_result_path is not None:
        save_dir = os.path.dirname(save_result_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_result_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved evaluation result to: {save_result_path}")

    return summary


# =========================================================
# 6. main
# =========================================================

if __name__ == "__main__":
    prompt_dir = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v5"
    save_result_path = "/home/wenbin.guo/DKGE4R/eval/test_prompts_v4_eval_result.json"

    evaluate_prompt_directory(
        prompt_dir=prompt_dir,
        save_result_path=save_result_path,
        model_path=MODEL_PATH,
        temperature=0.0,
        max_tokens=256,
        timeout=300,
        show_every=50,
        limit=None,      # 调试时可改成 100 / 500
        verbose=False
    )
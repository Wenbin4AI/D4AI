import os
import json
from collections import defaultdict


# =========================================================
# 1. 读取实体 / 关系本体
# =========================================================

def load_entities(entity_path):
    with open(entity_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entity_info = {}
    for x in data:
        eid = int(x["value"])
        entity_info[eid] = {
            "label": x.get("label", f"entity_{eid}"),
            "classname": x.get("classname", "Unknown"),
            "classid": x.get("classid", None),
            "freebase_id": x.get("freebase_id", "")
        }
    return entity_info


def load_relations(relation_path):
    with open(relation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    relation_info = {}
    for x in data:
        rid = int(x["id"])
        relation_info[rid] = {
            "label": x.get("label", f"relation_{rid}"),
            "domain": x.get("domain", "Unknown"),
            "range": x.get("range", "Unknown"),
            "domain_candidates": x.get("domain_candidates", []),
            "freebase": x.get("freebase", "")
        }
    return relation_info


# =========================================================
# 2. 基础工具函数
# =========================================================

def safe_entity_name(entity_info, eid):
    return entity_info.get(eid, {}).get("label", f"entity_{eid}")


def safe_entity_class(entity_info, eid):
    return entity_info.get(eid, {}).get("classname", "Unknown")


def safe_relation_name(relation_info, rid):
    return relation_info.get(rid, {}).get("label", f"relation_{rid}")


def normalize_path_score(path_score):
    try:
        return -float(path_score)
    except Exception:
        return -999999.0


def class_match(entity_class, expected_class):
    if expected_class is None or expected_class == "Unknown":
        return False
    return entity_class == expected_class


def unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================================================
# 3. 抽取紧凑型 query-level evidence
# =========================================================

def extract_compact_query_evidence(subgraph, entity_info, relation_info, max_path_keep=2, max_struct_keep=3):
    query = subgraph["query"]
    head = int(query["head"])
    relation = int(query["relation"])
    candidate_tails = [int(x) for x in query.get("candidate_tails", [])]

    head_label = safe_entity_name(entity_info, head)
    head_class = safe_entity_class(entity_info, head)

    rel_meta = relation_info.get(relation, {})
    r_label = rel_meta.get("label", f"relation_{relation}")
    r_domain = rel_meta.get("domain", "Unknown")
    r_range = rel_meta.get("range", "Unknown")

    edges = subgraph.get("edges", [])
    paths = subgraph.get("paths", [])
    key_nodes = set(subgraph.get("key_nodes", []))
    gold_tail = subgraph.get("gold_tail", None)
    if gold_tail is not None:
        gold_tail = int(gold_tail)

    direct_tails = []
    degree = defaultdict(int)
    aux_degree = defaultdict(int)

    for e in edges:
        h = int(e["head"])
        r = int(e["relation"])
        t = int(e["tail"])

        degree[h] += 1
        degree[t] += 1

        if h == head and r == relation:
            direct_tails.append(t)

        if r != relation:
            aux_degree[h] += 1
            aux_degree[t] += 1

    direct_tails = unique_keep_order(direct_tails)
    direct_tail_set = set(direct_tails)

    strong_direct_tails = sorted(
        direct_tails,
        key=lambda x: (
            int(class_match(safe_entity_class(entity_info, x), r_range)),
            int(x in key_nodes),
            aux_degree[x],
            degree[x]
        ),
        reverse=True
    )[:max_struct_keep]

    filtered_paths = []
    for p in paths:
        tail = int(p.get("tail"))
        path_nodes = [int(n) for n in p.get("nodes", [])]
        path_rels = [int(r) for r in p.get("relations", [])]
        score = normalize_path_score(p.get("path_score", -999999))

        if not path_nodes or path_nodes[0] != head:
            continue

        filtered_paths.append({
            "tail": tail,
            "nodes": path_nodes,
            "relations": path_rels,
            "score": score,
            "raw_score": p.get("path_score", None)
        })

    filtered_paths = sorted(
        filtered_paths,
        key=lambda x: (int(x["tail"] in direct_tail_set), x["score"]),
        reverse=True
    )
    best_paths = filtered_paths[:max_path_keep]

    evidence = []

    # 1. ontology
    text = f"Relation {r_label} expects tail type {r_range}"
    if r_domain != "Unknown":
        text += f" and head type {r_domain}"
    text += f". The head {head_label} is of class {head_class}."
    evidence.append(text)

    # 2. direct pattern
    evidence.append(
        f"There are {len(direct_tails)} direct candidates connected from {head_label} by relation {r_label}; direct target-relation support is a primary signal."
    )

    # 3. structural cluster
    if strong_direct_tails:
        names = []
        for t in strong_direct_tails:
            t_label = safe_entity_name(entity_info, t)
            t_class = safe_entity_class(entity_info, t)
            names.append(f"{t_label} [{t_class}]")
        evidence.append(
            f"Structurally strong direct candidates include: {', '.join(names)}."
        )

    # 4. path pattern
    if best_paths:
        path_descs = []
        for p in best_paths:
            node_names = [safe_entity_name(entity_info, n) for n in p["nodes"]]
            path_descs.append(" -> ".join(node_names))
        evidence.append(
            f"Useful supporting paths include: {'; '.join(path_descs)}."
        )

    # 5. summary
    evidence.append(
        "The best answer should match the expected tail type and also have strong direct or local structural support."
    )

    return {
        "head_id": head,
        "relation_id": relation,
        "gold_tail": gold_tail,
        "head_label": head_label,
        "head_class": head_class,
        "relation_label": r_label,
        "relation_domain": r_domain,
        "relation_range": r_range,
        "candidate_tails": candidate_tails,
        "query_evidence": evidence
    }


# =========================================================
# 4. 候选筛选：保证 gold 在 top-k
# =========================================================

def score_candidate_for_selection(
    tail_id,
    relation_range,
    direct_tail_set,
    key_nodes,
    degree_map,
    aux_degree_map,
    entity_info
):
    t_class = safe_entity_class(entity_info, tail_id)

    score = 0.0
    if class_match(t_class, relation_range):
        score += 4.0
    if tail_id in direct_tail_set:
        score += 5.0
    if tail_id in key_nodes:
        score += 2.0
    score += 0.3 * degree_map.get(tail_id, 0)
    score += 0.5 * aux_degree_map.get(tail_id, 0)
    return score


def select_topk_candidates(subgraph, entity_info, relation_info, topk=20):
    query = subgraph["query"]
    relation = int(query["relation"])
    head = int(query["head"])
    candidate_tails = [int(x) for x in query.get("candidate_tails", [])]
    candidate_tails = unique_keep_order(candidate_tails)

    gold_tail = subgraph.get("gold_tail", None)
    if gold_tail is not None:
        gold_tail = int(gold_tail)

    rel_meta = relation_info.get(relation, {})
    r_range = rel_meta.get("range", "Unknown")

    edges = subgraph.get("edges", [])
    key_nodes = set(subgraph.get("key_nodes", []))

    degree = defaultdict(int)
    aux_degree = defaultdict(int)
    direct_tail_set = set()

    for e in edges:
        h = int(e["head"])
        r = int(e["relation"])
        t = int(e["tail"])

        degree[h] += 1
        degree[t] += 1

        if h == head and r == relation:
            direct_tail_set.add(t)

        if r != relation:
            aux_degree[h] += 1
            aux_degree[t] += 1

    scored = []
    for tail_id in candidate_tails:
        s = score_candidate_for_selection(
            tail_id=tail_id,
            relation_range=r_range,
            direct_tail_set=direct_tail_set,
            key_nodes=key_nodes,
            degree_map=degree,
            aux_degree_map=aux_degree,
            entity_info=entity_info
        )
        scored.append((tail_id, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [x[0] for x in scored[:topk]]
    selected = unique_keep_order(selected)

    if gold_tail is not None and gold_tail in candidate_tails and gold_tail not in selected:
        if len(selected) < topk:
            selected.append(gold_tail)
        else:
            selected[-1] = gold_tail
        selected = unique_keep_order(selected)

        if len(selected) < topk:
            for tail_id, _ in scored:
                if tail_id not in selected:
                    selected.append(tail_id)
                if len(selected) >= topk:
                    break

    return selected[:topk]


# =========================================================
# 5. 生成候选行与 gold index（从0开始）
# =========================================================

def build_candidate_lines(selected_candidates, entity_info, gold_tail=None):
    candidate_lines = []
    gold_selected_index = None

    for idx, tail_id in enumerate(selected_candidates):
        label = safe_entity_name(entity_info, tail_id)
        classname = safe_entity_class(entity_info, tail_id)

        item = {
            "selected_index": idx,
            "tail_id": tail_id,
            "label": label,
            "classname": classname,
            "text": f"{idx}. {label} [class={classname}]"
        }
        candidate_lines.append(item)

        if gold_tail is not None and tail_id == gold_tail:
            gold_selected_index = idx

    return candidate_lines, gold_selected_index


# =========================================================
# 6. 构造单选 JSON prompt
# =========================================================

def build_single_choice_prompt(result, candidate_lines):
    head_label = result["head_label"]
    relation_label = result["relation_label"]
    evidence = result["query_evidence"]

    candidate_block = "\n".join([item["text"] for item in candidate_lines])
    evidence_block = "\n".join([f"- {x}" for x in evidence])

    prompt = f"""You are an expert knowledge graph completion system.

Your task is to select the correct tail entity from a candidate list.

Triple format:
(head entity, relation, tail entity)

Important:
- You are predicting ONLY the tail entity.
- The head entity "{head_label}" is NOT a valid answer.
- You MUST select exactly ONE candidate index.
- You MUST NOT output entity names.
- You MUST NOT generate new entities.
- Use the ontology and structural evidence to compare candidates.
- Output ONLY valid JSON.

Before answering, internally:
1. Understand the semantic meaning of the relation.
2. Determine the expected type of the tail entity.
3. Use the key evidence to compare all candidates carefully.
4. Select the most logically consistent candidate.
Do NOT output reasoning.

Key evidence:
{evidence_block}

Candidate entities:
{candidate_block}

Output format:
{{
  "selected_index": integer
}}

Incomplete triple:
({head_label}, {relation_label}, ?)
"""
    return prompt


# =========================================================
# 7. 处理单个子图文件
# =========================================================

def process_one_subgraph_file(subgraph_file, entity_info, relation_info, topk=20):
    with open(subgraph_file, "r", encoding="utf-8") as f:
        subgraph = json.load(f)

    result = extract_compact_query_evidence(subgraph, entity_info, relation_info)

    gold_tail = result["gold_tail"]
    selected_candidates = select_topk_candidates(
        subgraph=subgraph,
        entity_info=entity_info,
        relation_info=relation_info,
        topk=topk
    )

    candidate_lines, gold_selected_index = build_candidate_lines(
        selected_candidates=selected_candidates,
        entity_info=entity_info,
        gold_tail=gold_tail
    )

    prompt = build_single_choice_prompt(result, candidate_lines)

    return {
        "file": os.path.basename(subgraph_file),
        "head_id": result["head_id"],
        "relation_id": result["relation_id"],
        "gold_tail": gold_tail,
        "gold_selected_index": gold_selected_index,
        "head_label": result["head_label"],
        "head_class": result["head_class"],
        "relation_label": result["relation_label"],
        "relation_domain": result["relation_domain"],
        "relation_range": result["relation_range"],
        "query_evidence": result["query_evidence"],
        "candidate_lines": candidate_lines,
        "prompt": prompt
    }


# =========================================================
# 8. 批量处理并保存
# =========================================================

def process_and_save_per_file(
    subgraph_dir,
    entity_path,
    relation_path,
    output_dir,
    topk=20
):
    entity_info = load_entities(entity_path)
    relation_info = load_relations(relation_path)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output dir ready: {output_dir}")

    json_files = sorted([f for f in os.listdir(subgraph_dir) if f.endswith(".json")])
    print(f"[INFO] Found {len(json_files)} subgraph files")

    success_count = 0
    fail_count = 0
    missing_gold_count = 0

    for idx, fname in enumerate(json_files, start=1):
        fpath = os.path.join(subgraph_dir, fname)
        try:
            result = process_one_subgraph_file(
                subgraph_file=fpath,
                entity_info=entity_info,
                relation_info=relation_info,
                topk=topk
            )

            if result["gold_tail"] is not None and result["gold_selected_index"] is None:
                missing_gold_count += 1

            out_name = fname.replace(".json", "_prompt.json")
            out_path = os.path.join(output_dir, out_name)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            success_count += 1

            if idx % 50 == 0 or idx == 1 or idx == len(json_files):
                print(
                    f"[{idx}/{len(json_files)}] saved={out_name} | "
                    f"success={success_count} | fail={fail_count} | missing_gold={missing_gold_count}"
                )
        except Exception as e:
            fail_count += 1
            print(f"[ERROR] {fname} failed: {e}")

    print("==== Prompt Generation Finished ====")
    print(f"Total files    : {len(json_files)}")
    print(f"Success        : {success_count}")
    print(f"Fail           : {fail_count}")
    print(f"Missing gold   : {missing_gold_count}")


# =========================================================
# 9. main
# =========================================================

if __name__ == "__main__":
    entity_path = "/home/wenbin.guo/DKGE4R/data/FB15k-237/entity.json"
    relation_path = "/home/wenbin.guo/DKGE4R/data/FB15k-237/relation_new.json"

    subgraph_dir = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test"
    output_dir = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_v5"

    process_and_save_per_file(
        subgraph_dir=subgraph_dir,
        entity_path=entity_path,
        relation_path=relation_path,
        output_dir=output_dir,
        topk=20
    )
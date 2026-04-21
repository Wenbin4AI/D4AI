import os
import json
import math
from collections import defaultdict

# =========================================================
# 1. 读取本体信息
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
# 2. 工具函数
# =========================================================

def safe_entity_name(entity_info, eid):
    return entity_info.get(eid, {}).get("label", f"entity_{eid}")

def safe_entity_class(entity_info, eid):
    return entity_info.get(eid, {}).get("classname", "Unknown")

def safe_relation_name(relation_info, rid):
    return relation_info.get(rid, {}).get("label", f"relation_{rid}")

def class_match(entity_class, expected_class):
    if expected_class is None or expected_class == "Unknown":
        return 0
    if entity_class == expected_class:
        return 1
    # 可扩展：如果你之后有 subclass 信息，可在这里做软匹配
    return 0

def normalize_path_score(path_score):
    """
    你的 path_score 越大越好还是越接近 0 越好，目前样例里 -0.1 > -2.0。
    这里先做一个简单映射：score越接近0越高。
    """
    return -float(path_score)


# =========================================================
# 3. 为单个查询建立索引
# =========================================================

def build_subgraph_indexes(subgraph):
    edges = subgraph.get("edges", [])
    paths = subgraph.get("paths", [])
    key_nodes = set(subgraph.get("key_nodes", []))

    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    edge_set = set()

    for e in edges:
        h = int(e["head"])
        r = int(e["relation"])
        t = int(e["tail"])
        out_edges[h].append((r, t))
        in_edges[t].append((h, r))
        edge_set.add((h, r, t))

    paths_by_tail = defaultdict(list)
    for p in paths:
        tail = int(p["tail"])
        paths_by_tail[tail].append(p)

    return {
        "out_edges": out_edges,
        "in_edges": in_edges,
        "edge_set": edge_set,
        "paths_by_tail": paths_by_tail,
        "key_nodes": key_nodes,
    }


# =========================================================
# 4. 为候选生成证据
# =========================================================

def extract_candidate_evidence(
    head_id,
    relation_id,
    tail_id,
    subgraph_idx,
    entity_info,
    relation_info,
    top_aux_neighbors=3
):
    relation_meta = relation_info.get(relation_id, {})
    r_label = relation_meta.get("label", f"relation_{relation_id}")
    expected_domain = relation_meta.get("domain", "Unknown")
    expected_range = relation_meta.get("range", "Unknown")

    h_label = safe_entity_name(entity_info, head_id)
    h_class = safe_entity_class(entity_info, head_id)
    t_label = safe_entity_name(entity_info, tail_id)
    t_class = safe_entity_class(entity_info, tail_id)

    out_edges = subgraph_idx["out_edges"]
    in_edges = subgraph_idx["in_edges"]
    edge_set = subgraph_idx["edge_set"]
    paths_by_tail = subgraph_idx["paths_by_tail"]
    key_nodes = subgraph_idx["key_nodes"]

    evidence = []
    score = 0.0

    # ---------- 1) 本体约束 ----------
    h_domain_match = class_match(h_class, expected_domain)
    t_range_match = class_match(t_class, expected_range)

    evidence.append(
        f"[Ontology] Relation {r_label} expects head type {expected_domain} and tail type {expected_range}. "
        f"Head {h_label} is {h_class}; candidate {t_label} is {t_class}."
    )

    if h_domain_match:
        score += 1.5
    if t_range_match:
        score += 3.0
        evidence.append(
            f"[Type Filter] Candidate {t_label} matches the expected tail type {expected_range}."
        )
    else:
        evidence.append(
            f"[Type Filter] Candidate {t_label} does not clearly match the expected tail type {expected_range}."
        )

    # ---------- 2) 目标关系直连 ----------
    if (head_id, relation_id, tail_id) in edge_set:
        score += 4.0
        evidence.append(
            f"[Direct Edge] The subgraph contains a direct edge ({h_label}, {r_label}, {t_label})."
        )

    # ---------- 3) 路径模式 ----------
    candidate_paths = paths_by_tail.get(tail_id, [])
    if candidate_paths:
        best_path = max(candidate_paths, key=lambda p: normalize_path_score(p.get("path_score", -999)))
        rel_seq = [safe_relation_name(relation_info, r) for r in best_path.get("relations", [])]
        node_seq = [safe_entity_name(entity_info, n) for n in best_path.get("nodes", [])]
        best_score = normalize_path_score(best_path.get("path_score", -999))

        score += 1.5 + 0.2 * best_score
        evidence.append(
            f"[Path Pattern] Supporting path: {' -> '.join(node_seq)} "
            f"with relations {' -> '.join(rel_seq)}."
        )

    # ---------- 4) 局部结构 ----------
    aux_neighbors = []
    for r, t2 in out_edges.get(tail_id, []):
        if r != relation_id:
            aux_neighbors.append((r, t2))
    for h2, r in in_edges.get(tail_id, []):
        if r != relation_id:
            aux_neighbors.append((r, h2))

    score += min(len(aux_neighbors), 5) * 0.5

    if tail_id in key_nodes:
        score += 0.8
        evidence.append(
            f"[Key Node] Candidate {t_label} appears in the subgraph key node set."
        )

    if aux_neighbors:
        aux_neighbors = aux_neighbors[:top_aux_neighbors]
        aux_text = []
        for r, nid in aux_neighbors:
            aux_text.append(f"{safe_relation_name(relation_info, r)} -> {safe_entity_name(entity_info, nid)}")
        evidence.append(
            f"[Neighborhood] Candidate {t_label} has auxiliary structural support: " +
            "; ".join(aux_text) + "."
        )

    # ---------- 5) 综合摘要 ----------
    compact_summary = []
    if t_range_match:
        compact_summary.append("type-compatible")
    if (head_id, relation_id, tail_id) in edge_set:
        compact_summary.append("direct-target-edge")
    if candidate_paths:
        compact_summary.append("supporting-path")
    if tail_id in key_nodes:
        compact_summary.append("key-node")
    if aux_neighbors:
        compact_summary.append("dense-local-neighborhood")

    if compact_summary:
        evidence.append(
            "[Summary] " + t_label + " is supported by: " + ", ".join(compact_summary) + "."
        )

    return {
        "tail_id": tail_id,
        "tail_label": t_label,
        "tail_class": t_class,
        "score": round(score, 4),
        "evidence": evidence
    }


# =========================================================
# 5. 为整个查询生成“关键证据”
# =========================================================

def extract_query_level_evidence(subgraph, entity_info, relation_info, max_global_evidence=5):
    query = subgraph["query"]
    head_id = int(query["head"])
    relation_id = int(query["relation"])
    candidate_tails = [int(x) for x in query["candidate_tails"]]

    head_label = safe_entity_name(entity_info, head_id)
    head_class = safe_entity_class(entity_info, head_id)

    relation_meta = relation_info.get(relation_id, {})
    r_label = relation_meta.get("label", f"relation_{relation_id}")
    r_domain = relation_meta.get("domain", "Unknown")
    r_range = relation_meta.get("range", "Unknown")

    idx = build_subgraph_indexes(subgraph)

    # 为所有候选做局部证据抽取
    candidate_infos = []
    for tail_id in candidate_tails:
        candidate_infos.append(
            extract_candidate_evidence(
                head_id, relation_id, tail_id,
                idx, entity_info, relation_info
            )
        )

    candidate_infos.sort(key=lambda x: x["score"], reverse=True)

    global_evidence = []
    global_evidence.append(
        f"[Query Ontology] The query is ({head_label}, {r_label}, ?). "
        f"The relation expects domain {r_domain} and range {r_range}, while the head entity is of class {head_class}."
    )

    # 高分候选概览
    top_candidates = candidate_infos[:3]
    if top_candidates:
        tc_text = []
        for c in top_candidates:
            tc_text.append(f"{c['tail_label']} (score={c['score']}, class={c['tail_class']})")
        global_evidence.append(
            "[Top Structural Candidates] " + "; ".join(tc_text) + "."
        )

    # 统计直连候选数量
    direct_count = 0
    for c in candidate_infos:
        for ev in c["evidence"]:
            if ev.startswith("[Direct Edge]"):
                direct_count += 1
                break
    global_evidence.append(
        f"[Direct Support Pattern] {direct_count} candidate tails have direct target-relation edges from the head in the subgraph."
    )

    # 抽取最终精炼证据
    final_evidence = global_evidence[:]
    for c in candidate_infos[:3]:
        # 每个高分候选选2条最关键证据
        selected = []
        for ev in c["evidence"]:
            if ev.startswith("[Type Filter]") or ev.startswith("[Direct Edge]") or ev.startswith("[Path Pattern]") or ev.startswith("[Summary]"):
                selected.append(ev)
            if len(selected) >= 2:
                break
        final_evidence.extend(selected)

    return {
        "head_id": head_id,
        "relation_id": relation_id,
        "head_label": head_label,
        "relation_label": r_label,
        "query_evidence": final_evidence[:max_global_evidence],
        "candidate_details": candidate_infos
    }


# =========================================================
# 6. 生成可直接输入 LLM 的 prompt
# =========================================================

def build_prompt_from_evidence(result, candidate_topk=20):
    head_label = result["head_label"]
    relation_label = result["relation_label"]

    lines = []
    lines.append("You are given a knowledge graph link prediction task.")
    lines.append("")
    lines.append(f"Query: (head={head_label}, relation={relation_label}, tail=?)")
    lines.append("")
    lines.append("Key evidence:")
    for ev in result["query_evidence"]:
        lines.append(f"- {ev}")

    lines.append("")
    lines.append("Candidates:")
    for idx, c in enumerate(result["candidate_details"][:candidate_topk], start=1):
        lines.append(f"{idx}. {c['tail_label']} [class={c['tail_class']}]")

    lines.append("")
    lines.append("Instruction:")
    lines.append("Rank the candidate numbers in descending likelihood.")
    lines.append("Use ontology constraints, direct target-relation support, path evidence, and local neighborhood evidence jointly.")
    lines.append("Output only one Python-style list of candidate numbers.")
    lines.append("Do not explain.")

    return "\n".join(lines)


# =========================================================
# 7. 批量处理目录
# =========================================================

def process_subgraph_directory(subgraph_dir, entity_path, relation_path, output_path):
    entity_info = load_entities(entity_path)
    relation_info = load_relations(relation_path)

    results = []

    for fname in sorted(os.listdir(subgraph_dir)):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(subgraph_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            subgraph = json.load(f)

        result = extract_query_level_evidence(subgraph, entity_info, relation_info)
        prompt = build_prompt_from_evidence(result, candidate_topk=20)

        results.append({
            "file": fname,
            "head_id": result["head_id"],
            "relation_id": result["relation_id"],
            "head_label": result["head_label"],
            "relation_label": result["relation_label"],
            "query_evidence": result["query_evidence"],
            "candidate_details": result["candidate_details"],
            "prompt": prompt
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} query evidence results to: {output_path}")


# =========================================================
# 8. main
# =========================================================

if __name__ == "__main__":
    entity_path = "/path/to/entity.json"
    relation_path = "/path/to/relation.json"
    subgraph_dir = "/path/to/subgraph_dir"
    output_path = "/path/to/query_evidence_results.json"

    process_subgraph_directory(subgraph_dir, entity_path, relation_path, output_path)
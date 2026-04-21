import os
import json
import math
import random
from collections import defaultdict, Counter

# =========================
# 路径配置
# =========================
DATA_DIR = "/home/wenbin.guo/DKGE4R/data/FB15k-237"
SUBGRAPH_DIR = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test"
OUTPUT_DIR = "/home/wenbin.guo/DKGE4R/KGE_model/saved_subgraphs/test_prompts_hit1"

ENTITY_PATH = os.path.join(DATA_DIR, "entity.json")
RELATION_PATH = os.path.join(DATA_DIR, "relation_new.json")

CANDIDATE_LIMIT = 20
NEGATIVE_SIZE = CANDIDATE_LIMIT - 1
RANDOM_SEED = 42

# =========================
# 基础读写
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# =========================
# 映射
# =========================
def build_entity_map(entities):
    return {int(e["value"]): e for e in entities}

def build_relation_map(relations):
    return {int(r["id"]): r for r in relations}

def get_entity_label(entity_map, eid):
    return entity_map.get(eid, {}).get("label", f"Entity_{eid}")

def get_entity_class(entity_map, eid):
    return entity_map.get(eid, {}).get("classname", "Unknown")

def get_relation_label(relation_map, rid):
    return relation_map.get(rid, {}).get("label", f"Relation_{rid}")

# =========================
# 图结构处理
# =========================
def build_direct_edge_set(edges):
    return {(e["head"], e["relation"], e["tail"]) for e in edges}

def group_paths_by_tail(paths):
    tail2paths = defaultdict(list)
    for p in paths:
        tail2paths[p["tail"]].append(p)
    return tail2paths

def build_aux_neighbors(edges, target_relation):
    aux_out = defaultdict(list)
    aux_in = defaultdict(list)
    for e in edges:
        h, r, t = e["head"], e["relation"], e["tail"]
        if r != target_relation:
            aux_out[h].append((r, t))
            aux_in[t].append((r, h))
    return aux_out, aux_in

def count_shared_support(candidate, candidate_set, aux_out, aux_in):
    count = 0
    for _, nb in aux_out.get(candidate, []):
        if nb in candidate_set:
            count += 1
    for _, nb in aux_in.get(candidate, []):
        if nb in candidate_set:
            count += 1
    return count

def count_target_relation_edges_from_head(h, r, edges):
    return sum(1 for e in edges if e["head"] == h and e["relation"] == r)

# =========================
# 路径模式摘要
# =========================
def summarize_relation_path_patterns(paths, relation_map, topk=5):
    pattern_counter = Counter()
    for p in paths:
        rels = tuple(p.get("relations", []))
        if rels:
            pattern_counter[rels] += 1
    if not pattern_counter:
        return ["- No explicit path patterns were extracted."]
    lines = []
    for rel_seq, cnt in pattern_counter.most_common(topk):
        rel_labels = [get_relation_label(relation_map, rid) for rid in rel_seq]
        lines.append(f"- {' -> '.join(rel_labels)} (count={cnt})")
    return lines

# =========================
# 候选评分
# =========================
def get_range_candidates_set(rel_info, topk=5):
    result = set()
    for item in rel_info.get("range_candidates", [])[:topk]:
        cls = item.get("classname", "Unknown")
        if cls != "Unknown":
            result.add(cls)
    return result

def compute_candidate_features(
    t, h, r, rel_info, entity_map, direct_edges,
    tail2paths, aux_out, aux_in, key_nodes, candidate_set
):
    expected_range = rel_info.get("range", "Unknown")
    top_range_classes = get_range_candidates_set(rel_info)

    t_class = get_entity_class(entity_map, t)
    type_match = int(expected_range != "Unknown" and t_class == expected_range)
    soft_type_match = int(t_class in top_range_classes)
    direct = int((h, r, t) in direct_edges)
    path_count = len(tail2paths.get(t, []))
    aux_support = count_shared_support(t, candidate_set, aux_out, aux_in)
    key_flag = int(t in key_nodes)

    score = (
        3.0*type_match + 1.5*soft_type_match + 3.0*direct +
        1.5*math.log1p(path_count) + 1.0*math.log1p(aux_support) + 1.0*key_flag
    )

    return {
        "candidate_id": t,
        "candidate_label": get_entity_label(entity_map, t),
        "candidate_class": t_class,
        "features": {
            "type_match": type_match,
            "soft_type_match": soft_type_match,
            "direct": direct,
            "path_count": path_count,
            "aux_support": aux_support,
            "key": key_flag
        },
        "score": round(score, 6)
    }

def select_candidates_with_single_gold(
    candidate_tails, gold_tail, h, r, rel_info,
    entity_map, direct_edges, tail2paths, aux_out, aux_in, key_nodes,
    candidate_limit=20, random_seed=42
):
    candidate_set_full = set(candidate_tails)
    all_infos = [compute_candidate_features(
        t, h, r, rel_info, entity_map, direct_edges, tail2paths, aux_out, aux_in, key_nodes, candidate_set_full
    ) for t in candidate_tails]

    gold_info = None
    negatives = []
    for info in all_infos:
        if info["candidate_id"] == gold_tail:
            gold_info = info
        else:
            negatives.append(info)
    if gold_info is None:
        raise ValueError(f"gold_tail={gold_tail} not found in candidate_tails")
    rng = random.Random(random_seed)
    selected_negatives = rng.sample(negatives, candidate_limit-1)
    selected = [gold_info] + selected_negatives
    rng.shuffle(selected)
    return selected

# =========================
# Prompt构建
# =========================
def make_ontology_summary(h, r, entity_map, relation_map):
    rel_info = relation_map.get(r, {})
    h_label = get_entity_label(entity_map, h)
    h_class = get_entity_class(entity_map, h)
    r_label = get_relation_label(relation_map, r)
    domain = rel_info.get("domain", "Unknown")
    range_ = rel_info.get("range", "Unknown")
    lines = [
        f"- Query relation: {r_label}",
        f"- Head entity: {h_label}",
        f"- Head class: {h_class}",
        f"- Relation domain: {domain}",
        f"- Relation range: {range_}",
    ]
    if domain != "Unknown":
        lines.append(f"- Domain check: {'matched' if h_class==domain else 'not clearly matched'}")
    if range_ != "Unknown":
        lines.append(f"- Tail entities of class '{range_}' should be preferred.")
    return "\n".join(lines)

def make_subgraph_summary(h, r, subgraph, relation_map, entity_map):
    target_rel_count = count_target_relation_edges_from_head(h, r, subgraph["edges"])
    key_nodes = subgraph.get("key_nodes", [])
    lines = [
        f"- Local subgraph: nodes={len(subgraph.get('nodes',[]))}, edges={len(subgraph.get('edges',[]))}, paths={len(subgraph.get('paths',[]))}",
        f"- Head outgoing edges under target relation: {target_rel_count}"
    ]
    if key_nodes:
        key_labels = [get_entity_label(entity_map, x) for x in key_nodes[:8]]
        lines.append(f"- Key support nodes: {', '.join(key_labels)}")
    return "\n".join(lines)

def make_candidate_compact_line(idx, info):
    f = info["features"]
    return (
        f"{idx}. {info['candidate_label']} [id={info['candidate_id']} | class={info['candidate_class']} "
        f"| type={f['type_match']} | soft_type={f['soft_type_match']} | direct={f['direct']} "
        f"| paths={f['path_count']} | aux={f['aux_support']} | key={f['key']}]"
    )

def build_hit1_prompt(subgraph, entity_map, relation_map, candidate_limit=20, random_seed=42):
    h = subgraph["query"]["head"]
    r = subgraph["query"]["relation"]
    gold_tail = subgraph["gold_tail"]
    direct_edges = build_direct_edge_set(subgraph["edges"])
    tail2paths = group_paths_by_tail(subgraph.get("paths", []))
    aux_out, aux_in = build_aux_neighbors(subgraph["edges"], target_relation=r)
    key_nodes = set(subgraph.get("key_nodes", []))
    rel_info = relation_map.get(r, {"label": f"Relation_{r}"})

    selected_infos = select_candidates_with_single_gold(
        subgraph["query"]["candidate_tails"], gold_tail, h, r, rel_info,
        entity_map, direct_edges, tail2paths, aux_out, aux_in, key_nodes,
        candidate_limit, random_seed
    )

    candidate_lines = [make_candidate_compact_line(idx+1, info) for idx, info in enumerate(selected_infos)]
    candidate_id_to_rank_index = {info["candidate_id"]: idx+1 for idx, info in enumerate(selected_infos)}
    gold_rank_index = candidate_id_to_rank_index[gold_tail]

    ontology_summary = make_ontology_summary(h, r, entity_map, relation_map)
    subgraph_summary = make_subgraph_summary(h, r, subgraph, relation_map, entity_map)
    path_pattern_lines = summarize_relation_path_patterns(subgraph.get("paths", []), relation_map)

    prompt_text = f"""You are given a knowledge graph link prediction ranking task.

Your task is to select exactly ONE correct tail entity from the candidate list.

Head: {get_entity_label(entity_map, h)}
Relation: {get_relation_label(relation_map, r)}

Ontology summary:
{ontology_summary}

Local subgraph summary:
{subgraph_summary}

Frequent path patterns:
{chr(10).join(path_pattern_lines)}

Candidates:
{chr(10).join(candidate_lines)}

Output format:
{{"selected_index": integer}}
"""
    return {
        "prompt": prompt_text,
        "query": {
            "head_id": h,
            "relation_id": r,
            "gold_tail": gold_tail,
            "gold_candidate_rank_index": gold_rank_index,
            "selected_candidates": selected_infos
        }
    }

# =========================
# 文件处理
# =========================
def process_one_file(subgraph_path, entity_map, relation_map, output_dir, candidate_limit=20, random_seed=42):
    subgraph = load_json(subgraph_path)
    file_seed = random_seed + abs(hash(os.path.basename(subgraph_path))) % 1000000
    result = build_hit1_prompt(subgraph, entity_map, relation_map, candidate_limit, file_seed)

    filename = os.path.basename(subgraph_path)
    stem = os.path.splitext(filename)[0]
    os.makedirs(output_dir, exist_ok=True)

    json_out = os.path.join(output_dir, f"{stem}_prompt.json")
    txt_out = os.path.join(output_dir, f"{stem}_prompt.txt")

    save_json(result, json_out)
    save_text(result["prompt"], txt_out)

    print(f"Saved: {json_out}")
    print(f"Saved: {txt_out}")

def process_all_files():
    random.seed(RANDOM_SEED)
    entities = load_json(ENTITY_PATH)
    relations = load_json(RELATION_PATH)
    entity_map = build_entity_map(entities)
    relation_map = build_relation_map(relations)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [os.path.join(SUBGRAPH_DIR, x) for x in os.listdir(SUBGRAPH_DIR) if x.endswith(".json")]
    files.sort()
    for path in files:
        try:
            process_one_file(path, entity_map, relation_map, OUTPUT_DIR, CANDIDATE_LIMIT, RANDOM_SEED)
        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")

if __name__ == "__main__":
    process_all_files()
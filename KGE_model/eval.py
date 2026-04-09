import os
import json
import torch
from tqdm import tqdm

from model import KGEModel
from run import parse_args, override_config


# =========================
# ✔ 正确读取训练ID映射（关键修复）
# 如果你的数据是 entity2id.txt / relation2id.txt 格式就用这个
# =========================
def load_id_map(file_path):
    id_map = {}
    with open(file_path, "r") as f:
        for line in f:
            name, idx = line.strip().split("\t")
            id_map[name] = int(idx)
    return id_map


# =========================
# build mapping（修复版：优先用训练ID）
# =========================
def build_mapping(data_path):

    entity_file = os.path.join(data_path, "entity2id.txt")
    relation_file = os.path.join(data_path, "relation2id.txt")

    # ✔ 如果存在标准OpenKE格式，直接用（强烈推荐）
    if os.path.exists(entity_file) and os.path.exists(relation_file):
        print("[INFO] Using entity2id.txt / relation2id.txt (SAFE MODE)")
        entity2id = load_id_map(entity_file)
        relation2id = load_id_map(relation_file)
        return entity2id, relation2id

    # ❗ fallback（你原来的方式）
    print("[WARN] No id files found, fallback to rebuild mapping (RISK OF MISMATCH)")

    entity2id = {}
    relation2id = {}

    def scan(file):
        with open(file, "r") as f:
            for line in f:
                h, r, t = line.strip().split("\t")
                if h not in entity2id:
                    entity2id[h] = len(entity2id)
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
                if r not in relation2id:
                    relation2id[r] = len(relation2id)

    scan(os.path.join(data_path, "train.txt"))
    scan(os.path.join(data_path, "valid.txt"))
    scan(os.path.join(data_path, "test.txt"))

    return entity2id, relation2id


# =========================
# load model（修复 + check）
# =========================
def load_model(args):

    config_path = os.path.join(args.init_checkpoint, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    args.nentity = config["nentity"]
    args.nrelation = config["nrelation"]
    args.hidden_dim = config["hidden_dim"]
    args.gamma = config["gamma"]
    args.double_entity_embedding = config["double_entity_embedding"]
    args.double_relation_embedding = config["double_relation_embedding"]

    print("\n[DEBUG]")
    print("nentity =", args.nentity)
    print("nrelation =", args.nrelation, "\n")

    model = KGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )

    ckpt_path = os.path.join(args.init_checkpoint, "checkpoint")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("[INFO] Model loaded successfully ✔")

    return model


# =========================
# read triples（带进度）
# =========================
def read_triples(file, entity2id, relation2id):
    triples = []
    with open(file, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Loading {os.path.basename(file)}"):
        h, r, t = line.strip().split("\t")
        triples.append((entity2id[h], relation2id[r], entity2id[t]))

    return triples


# =========================
# main
# =========================
def main():

    args = parse_args()
    override_config(args)

    data_path = args.data_path

    # =========================
    # mapping
    # =========================
    entity2id, relation2id = build_mapping(data_path)

    # =========================
    # triples
    # =========================
    test_triples = read_triples(
        os.path.join(data_path, "test.txt"),
        entity2id,
        relation2id
    )

    train_triples = read_triples(
        os.path.join(data_path, "train.txt"),
        entity2id,
        relation2id
    )

    valid_triples = read_triples(
        os.path.join(data_path, "valid.txt"),
        entity2id,
        relation2id
    )

    all_true_triples = train_triples + valid_triples + test_triples

    # =========================
    # model
    # =========================
    model = load_model(args)

    if args.cuda:
        model = model.cuda()

    # =========================
    # eval（带进度提示）
    # =========================
    print("\n[INFO] Start evaluation ...\n")

    metrics = model.test_step(
        model,
        test_triples,
        all_true_triples,
        args
    )

    print("\n===== FINAL RESULT =====")
    print(metrics)


if __name__ == "__main__":
    main()
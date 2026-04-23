import json
import requests

# =========================
# 配置
# =========================
INPUT_FILE = "/home/wenbin.guo/DKGE4R/guijidian.txt"
API_URL = "http://124.223.143.72:8071/prune"
DISTANCE = 50  # 距离阈值


# =========================
# 读取数据
# =========================
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 调用接口
# =========================
def call_api(data, distance):
    payload = {
        "data": data,
        "distance": distance
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        print("请求失败:", response.status_code, response.text)
        return None

    return response.json()


# =========================
# 主函数
# =========================
if __name__ == "__main__":
    # 1. 读取文件
    data = load_data(INPUT_FILE)

    trajectory = data.get("data", {}).get("records", [])
    print(f"原始点数: {len(trajectory)}")

    # 2. 调用接口
    result = call_api(data, DISTANCE)

    if result is None:
        exit(1)

    # 3. 输出结果
    print("\n接口返回结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 4. 额外信息
    if "pruned_count" in result:
        print(f"\n剪枝后点数: {result['pruned_count']}")
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:22014/v1",
    api_key="EMPTY"   # vLLM 允许随便填
)

resp = client.chat.completions.create(
    model="/home/wenbin.guo/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Hello everone!"}]
)

print(resp.choices[0].message.content)
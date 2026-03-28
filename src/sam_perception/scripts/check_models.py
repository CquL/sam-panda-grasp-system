# -*- coding: utf-8 -*-
import os
from openai import OpenAI

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")

# 将 OpenAI 的客户端地址指向阿里云百炼的兼容接口
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

print("正在向阿里云百炼请求可用模型列表...\n")
try:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("请先设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    # 调用标准接口获取模型列表
    models_page = client.models.list()
    
    count = 0
    vl_models = []
    
    print("🌟 完整可用模型列表：")
    for model in models_page.data:
        print(f" - {model.id}")
        count += 1
        # 顺便把我们做机械臂需要的视觉多模态模型筛选出来
        if 'vl' in model.id.lower():
            vl_models.append(model.id)
            
    print(f"\n✅ 成功获取！当前账号可用模型总数: {count} 个。")
    
    print("\n==========================================")
    print("🤖 针对你的【机械臂视觉抓取项目】，建议使用以下多模态模型：")
    for vl in vl_models:
        print(f" 👉 {vl}")
    print("==========================================")
    
except Exception as e:
    print(f"❌ 获取列表失败: {e}")
    print("请检查 API Key 是否正确，或网络是否通畅。")

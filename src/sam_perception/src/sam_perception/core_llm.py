import os
from openai import OpenAI
import json

class LLMPlannerCore:
    def __init__(self, api_key=None, model="gpt-4o"): # 推荐用 4o 或者 qwen-vl
        """初始化大模型客户端"""
        # 如果不传 api_key，就默认从环境变量找
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def parse_user_command_to_bbox(self, user_command, scene_context=""):
        """
        将自然语言指令解析为目标物体的 BBox
        (这里是一个简化版的文本推理示例，如果用 VLM 传入图像也是类似逻辑)
        """
        system_prompt = """
        你是一个机器人视觉助手。你需要根据用户的指令，在当前场景中找到目标物体，
        并返回该物体的边界框坐标。
        请严格以 JSON 格式返回: {"target": "物体名称", "bbox": [x_min, y_min, x_max, y_max]}
        """
        
        user_prompt = f"场景信息: {scene_context}\n用户指令: {user_command}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={ "type": "json_object" }, # 强制返回 JSON
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            # 解析结果
            result_str = response.choices[0].message.content
            result_json = json.loads(result_str)
            return result_json['bbox']
            
        except Exception as e:
            print(f"[LLMPlannerCore] API 调用失败: {e}")
            return None
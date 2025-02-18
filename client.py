from openai import OpenAI
from typing import List, Optional, Any, Dict
import requests
import json

class LlamaClient(OpenAI):
    """自定义 OpenAI 客户端以支持 llama.cpp-server"""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        super().__init__(
            base_url=base_url, api_key="not-needed"  # llama.cpp-server 不需要 API key
        )


    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = 0.8,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        data = {
            "messages": messages,
            "temperature": temperature,
            "stop": stop or ["user:","assistant:","Human:", "Assistant:","  .","\n\n\n"],
        }
        if max_tokens:
            data['max_tokens']=max_tokens
        response = requests.post(f"{self.base_url}/chat/completions", json=data)
        response.raise_for_status()
        completion = response.json()

        return completion

# 使用示例
def main():
    # 创建客户端实例
    client = LlamaClient(base_url="http://127.0.0.1:8081")

    messages = [
            {
                'role': 'system', 
                'content': 'Make sure to state your answer and your confidence at the end of the response following format strictly.You should state your answer following this format:\nMy answer is *your answer*\nFor example, you must say in this format:My answer is 100.\nYou must follow this format to state your confidence is a float in [0,1] with at most two digits:\nFor example, you can say, my confidence is 0.85.\n'
            },
            {
                'role': 'user', 
                'content': 'What is the result of 52+77*46+60-69*26?'
            }
        ]
    completion = client.create_chat_completion(
        messages=messages, max_tokens=1280, temperature=0
    )
    print("\n=== Chat Completion 示例 ===")
    print(f'completion\n{json.dumps(completion,indent=2)}')
    print("\n=== message 示例 ===")
    print(completion["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()

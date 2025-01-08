from openai import OpenAI
from typing import List, Optional, Any, Dict
import requests


class LlamaClient(OpenAI):
    """自定义 OpenAI 客户端以支持 llama.cpp-server"""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        super().__init__(
            base_url=base_url, api_key="not-needed"  # llama.cpp-server 不需要 API key
        )

    def chat_completion_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将 ChatCompletion 格式的消息转换为普通文本提示词"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            prompt += f"{role}: {content}\n"
        return prompt + "assistant:"

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """创建文本补全"""
        data = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop or ["\n", "Human:", "Assistant:"],
        }

        response = requests.post(f"{self.base_url}/completion", json=data)
        response.raise_for_status()

        completion = response.json()
        return {
            "choices": [
                {
                    "text": completion.get("content", ""),
                    "finish_reason": (
                        "stop" if completion.get("stopped_eos", False) else "length"
                    ),
                }
            ]
        }

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.8,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """创建对话补全"""
        prompt = self.chat_completion_to_prompt(messages)
        completion = self.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": completion["choices"][0]["text"].strip(),
                    },
                    "finish_reason": completion["choices"][0]["finish_reason"],
                }
            ]
        }


# 使用示例
def main():
    # 创建客户端实例
    client = LlamaClient()

    # 示例 1: 使用 completion 接口
    completion = client.create_completion(
        prompt="Human: 你好,请介绍一下自己\nAssistant:", max_tokens=128, temperature=0.7
    )
    print("\n=== Completion 示例 ===")
    print(completion["choices"][0]["text"])

    # 示例 2: 使用 chat completion 接口
    messages = [{"role": "user", "content": "你好,请介绍一下自己"}]
    chat_completion = client.create_chat_completion(
        messages=messages, max_tokens=128, temperature=0.7
    )
    print("\n=== Chat Completion 示例 ===")
    print(chat_completion["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()

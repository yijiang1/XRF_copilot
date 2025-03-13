import os, base64, json, requests
from openai import OpenAI, NOT_GIVEN
from nodeology.client import LLM_Client, VLM_Client
from nodeology.node import remove_markdown_blocks_formatting

# ARGO Instructions:
# ==================
# export ANL_ID=your_anl_id
# export ARGO_ENDPOINT=the_argo_endpoint
ARGO_MODELS = ["gpt4o", "gpto1preview"]
ARGO_MODEL_OPTIONS = ["stop", "temperature", "top_p"]
ANL_ID = os.environ.get("ANL_ID")
ARGO_ENDPOINT = os.environ.get("ARGO_ENDPOINT")

# ALCF Instructions:
# https://github.com/argonne-lcf/inference-endpoints
# ==================
# ** First time setup:
# pip install globus_sdk
# pip install openai
# wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py
# python inference_auth_token.py authenticate
# -----------------
# ** Update expired token (after 48 hours):
# python inference_auth_token.py get_access_token
# -----------------
# ** Re-authenticate:
# python inference_auth_token.py authenticate --force
# -----------------

ALCF_MODELS = {
    "Qwen/QwQ-32B-Preview",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "Qwen/Qwen2-VL-72B-Instruct",
}


class Argo_Client(LLM_Client):
    def __init__(self, model_name, model_options=None) -> None:
        super().__init__()
        assert ARGO_ENDPOINT is not None, "ARGO_ENDPOINT is not set"
        assert ANL_ID is not None, "ANL_ID is not set"
        assert model_name in ARGO_MODELS, f"Model {model_name} is not supported"
        if model_options is not None:
            assert all(
                option in ARGO_MODEL_OPTIONS for option in model_options
            ), f"Model options {model_options} are not supported"
        else:
            model_options = {}
        self.url = ARGO_ENDPOINT
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "text/plain",  # Expect standard text streaming
            "Accept-Encoding": "identity",  # Prevent gzip compression
        }
        self.model_name = model_name
        self.model_options = model_options

    def __call__(self, messages, **kwargs) -> str:
        assert (
            isinstance(messages, list) and len(messages) > 0
        ), "messages must be a list with at least one message"

        if self.model_name == "gpt4o":
            input_messages = [
                {
                    "role": "system",
                    "content": kwargs.get(
                        "system", "You are an intelligent assistant with the name Argo."
                    ),
                }
            ] + messages
        elif self.model_name == "gpto1preview":
            input_messages = messages

        data = {
            "user": ANL_ID,
            "model": self.model_name,
            "messages": input_messages,
            "stop": self.model_options.get("stop", []),
            "temperature": self.model_options.get("temperature", 0.1),
            "top_p": self.model_options.get("top_p", 0.9),
        }

        payload = json.dumps(data)
        response = requests.post(self.url, data=payload, headers=self.headers)

        if response.status_code == 200:
            return remove_markdown_blocks_formatting(response.json()["response"])
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")


class ALCF_Client(VLM_Client):
    def __init__(
        self,
        model_name,
        model_options=None,
    ) -> None:
        super().__init__()
        assert model_name in ALCF_MODELS, f"Model {model_name} is not supported"
        from inference_auth_token import get_access_token

        try:
            ALCF_ACCESS_TOKEN = get_access_token()
        except Exception as e:
            print(f"Error: {e}")
            raise Exception("Failed to get ALCF access token")
        assert (
            ALCF_ACCESS_TOKEN is not None and len(ALCF_ACCESS_TOKEN) > 0
        ), "ALCF access token is not properly set"
        self.client = OpenAI(
            api_key=ALCF_ACCESS_TOKEN,
            base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
        )
        self.model_name = model_name
        self.model_options = model_options if model_options is not None else {}

    def process_images(self, messages, images, **kwargs) -> list:
        image_messages = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(open(img, 'rb').read()).decode('utf-8')}"
                },
            }
            for img in images
        ]
        if isinstance(messages[-1]["content"], str):
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}
            ] + image_messages
        elif isinstance(messages[-1]["content"], list):
            messages[-1]["content"] += image_messages
        return messages

    def __call__(self, messages, images=None, format=None) -> str:
        if images is not None:
            messages = self.process_images(messages, images)

        for _ in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=(
                        {"type": "json_object"} if format == "json" else NOT_GIVEN
                    ),
                    **self.model_options if images is None else {},
                )
                if format == "json":
                    test_json = json.loads(
                        remove_markdown_blocks_formatting(
                            completion.choices[0].message.content
                        )
                    )
                return remove_markdown_blocks_formatting(
                    completion.choices[0].message.content
                )
            except Exception as e:
                print(f"Error: {e}")
                continue
        raise ValueError("Failed to generate response")


if __name__ == "__main__":
    test_conversation = [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I am Mario."},
        {"role": "user", "content": "No your name is Argo."},
    ]
    argo_client = Argo_Client("gpt4o")
    print(argo_client(test_conversation))

    test_conversation = [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "I am Mario."},
        {"role": "user", "content": "No your name is Argo."},
    ]
    alcf_client = ALCF_Client("Qwen/QwQ-32B-Preview")
    print(alcf_client(test_conversation))

    test_message = {"role": "user", "content": "What is in the image?"}
    alcf_client = ALCF_Client("Qwen/Qwen2-VL-72B-Instruct")
    print("ALCF: ", alcf_client([test_message], images=["test.jpg"]))
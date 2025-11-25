from dataclasses import dataclass

@dataclass
class Config:
    # if you use Qwen
    OPENAI_API_KEY: str = 'your_api_key_here'
    OPENAI_BASE_URL: str = 'your_base_url_here'
    # DEFAULT_MODEL: str = "qwen-vl-plus-2025-01-25"
    DEFAULT_MODEL: str = "qvq-max"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TEMPERATURE: float = 0.7
    MAX_RETRIES: int = 3


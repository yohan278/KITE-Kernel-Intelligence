"""Model loaders for extracting ModelSpec from model configurations."""

from dataset_generator.model_loader.base import BaseModelLoader
from dataset_generator.model_loader.hf_loader import HuggingFaceModelLoader
from dataset_generator.model_loader.qwen3 import Qwen3ModelLoader
from dataset_generator.model_loader.gpt_oss import GPTOSSModelLoader
from dataset_generator.model_loader.glm import GLMModelLoader
from dataset_generator.model_loader.kimi import KimiModelLoader

__all__ = [
    "BaseModelLoader",
    "GLMModelLoader",
    "GPTOSSModelLoader",
    "HuggingFaceModelLoader",
    "KimiModelLoader",
    "Qwen3ModelLoader",
]

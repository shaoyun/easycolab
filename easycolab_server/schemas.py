"""EasyColab 服务端模块: schemas."""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ==============================================================================
# 供 macOS 原生客户端调用的异步 REST API 接口 (/minimax_api/* · 多模型通用)
# ==============================================================================
class SelectModelRequest(BaseModel):
    model_id: str = "minimax_h3"
    repo_id: str = ""
    include_pattern: str = ""
    custom_name: str = ""
    modality: str = "auto"
    hf_token: str = ""

class DownloadModelRequest(BaseModel):
    model_id: Optional[str] = None
    hf_token: str = ""
    repo_id: str = ""
    include_pattern: str = ""
    custom_name: str = ""
    modality: str = "auto"

class LoadModelRequest(BaseModel):
    model_id: Optional[str] = None
    precision_mode: str = "bf16"
    offload_mode: str = "disk"
    hf_token: str = ""
    context_length: Optional[int] = None
    repo_id: str = ""
    include_pattern: str = ""
    custom_name: str = ""
    modality: str = "auto"

class ContextConfigRequest(BaseModel):
    context_length: int = 32768

class IdleConfigRequest(BaseModel):
    idle_timeout_minutes: int = 30

class ComfyStartRequest(BaseModel):
    unload_existing: bool = False
    force_restart: bool = False

class ComfyWeightDownloadRequest(BaseModel):
    url_or_repo: str
    filename: str = ""
    subfolder: str = "diffusion_models"
    hf_token: str = ""

class ComfyInstallPackRequest(BaseModel):
    pack_id: str
    auto_start: bool = True
    hf_token: str = ""

class ToyboxPrepareRequest(BaseModel):
    include_rewriter: bool = True
    include_image: bool = True
    auto_start: bool = False
    hf_token: str = ""

class ToyboxStartRequest(BaseModel):
    unload_existing: bool = True
    force_restart: bool = False
    hf_token: str = ""
    save_api_keys: bool = False

class ToyboxStopRequest(BaseModel):
    stop_comfyui_too: bool = False

class GenerateVideoRequest(BaseModel):
    model_id: Optional[str] = None
    task_mode: str = "t2va"
    visual_prompt: str
    negative_prompt: str = ""
    soundscape_prompt: str = ""
    music_prompt: str = ""
    first_frame_base64: Optional[str] = None
    last_frame_base64: Optional[str] = None
    ref_images_base64: Optional[List[str]] = None
    ref_videos_base64: Optional[List[str]] = None
    ref_audios_base64: Optional[List[str]] = None
    width: int = 832
    height: int = 480
    num_frames: int = 124
    num_inference_steps: int = 35
    fps: int = 24
    seed: int = 42
    randomize_seed: bool = True
    batch_count: int = 1
    denoising_strength: float = 0.70
    enable_thinking: bool = True
    max_new_tokens: int = 16384
    context_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: Optional[float] = None
    repeat_penalty: float = 1.0
    reasoning_effort: str = "medium"
    chat_messages: Optional[List[Dict[str, str]]] = None

class ApiKeyConfigRequest(BaseModel):
    api_key: str = ""
    initialize_only: bool = False

class UnloadModelRequest(BaseModel):
    model_id: str = ""

class RegisterCustomModelRequest(BaseModel):
    repo_id: str
    include_pattern: str = ""
    custom_name: str = ""
    modality: str = "auto"
    modality_hint: str = "auto"
    hf_token: str = ""
    auto_download: bool = False

class RemoveCustomModelRequest(BaseModel):
    model_id: str
    delete_files: bool = False

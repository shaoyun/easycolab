"""EasyColab 服务端模块化核心包。"""

from . import runtime_state  # noqa: F401
from . import schemas  # noqa: F401
from . import qwen_kernels  # noqa: F401
from . import model_catalog  # noqa: F401
from . import llm_runtime  # noqa: F401
from . import media_pipeline  # noqa: F401
from . import comfyui_service  # noqa: F401
from . import toybox_service  # noqa: F401
from . import telemetry_service  # noqa: F401
from . import status_service  # noqa: F401
from . import openai_gateway  # noqa: F401
from . import macos_routes  # noqa: F401
from . import gradio_ui  # noqa: F401

_EXPORT_MODULES = (
    runtime_state,
    schemas,
    qwen_kernels,
    model_catalog,
    llm_runtime,
    media_pipeline,
    comfyui_service,
    toybox_service,
    telemetry_service,
    status_service,
    openai_gateway,
    macos_routes,
    gradio_ui,
)

__all__ = []
for _mod in _EXPORT_MODULES:
    for _name in dir(_mod):
        if _name.startswith("__"):
            continue
        globals()[_name] = getattr(_mod, _name)
        if _name not in __all__:
            __all__.append(_name)

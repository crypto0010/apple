"""Model implementations.

Eagerly imports every model module so that their ``@registry.register(...)``
decorators run at package import time. Without this, the registry stays empty
unless callers happen to import each model module themselves (which the unit
tests do, but the train-all driver did not).
"""

from apple_bench.models import (  # noqa: F401  -- import-for-side-effects (registry)
    efficientnet_lite,
    mobilenet_v3,
    paper1_aco_svm,
    paper2_econv_vit,
    paper3_efficientnet,
    paper4_resnet,
    yolov8_cls,
)

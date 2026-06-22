import os
import warnings
from collections.abc import Mapping

__all__ = ("ModelDeprecationWarning", "warn_if_deprecated")

# Frames under the package are skipped so the warning points at the caller.
_PKG_PREFIX = os.path.dirname(__file__) + os.sep


class ModelDeprecationWarning(FutureWarning):
    """A model passed to a client is scheduled for retirement by its provider.

    Subclasses ``FutureWarning`` so it surfaces to end users by default. The
    deprecation comes from the provider's own metadata, not from this library, and
    the model keeps working until its retirement date — so this is advisory and
    never raises. Silence a deliberately pinned model with
    ``warnings.filterwarnings("ignore", category=ModelDeprecationWarning)``.
    """


# (provider, model) pairs already warned about, to warn once per process.
_warned: set[tuple[str, str]] = set()


def warn_if_deprecated(
    provider: str, model: str | None, deprecations: Mapping[str, str]
) -> None:
    """Emit a one-time warning when *model* is a provider-deprecated id.

    *deprecations* maps a still-served model id to its ISO-8601 retirement date,
    sourced from the weekly model-drift refresh. Unknown models and a ``None``
    *model* no-op. The warning fires once per ``(provider, model)`` pair per process
    to avoid fatigue and is attributed to the first caller outside this package.
    """
    if model is None:
        return
    retirement = deprecations.get(model)
    if retirement is None:
        return
    key = (provider, model)
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(
        f"{provider} model {model!r} is scheduled for retirement on {retirement}; "
        "migrate to a supported model before then.",
        ModelDeprecationWarning,
        skip_file_prefixes=(_PKG_PREFIX,),
    )

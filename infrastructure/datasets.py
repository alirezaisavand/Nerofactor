"""Opt-in dataset construction; existing training entry points remain legacy."""

from infrastructure.cache import PreprocessingCache
from infrastructure.config import ConfigError


def load_database(configuration, resolver, run):
    if configuration.schema_version != 1 or configuration.inputs is None:
        raise ConfigError("Isolated dataset loading requires a schema 1 envelope")
    if any(d.level == "error" for d in configuration.diagnostics):
        raise ConfigError("Resolve configuration errors before isolated dataset loading")
    selected = resolver.resolve_inputs(configuration.inputs)
    cache = PreprocessingCache(resolver, run, configuration.preprocessing)
    # Import only when dataset construction is explicitly requested. The config
    # and storage foundations continue to work without scientific dependencies.
    from dataset.database import parse_database_name
    return parse_database_name(configuration.legacy["database_name"], str(selected.dataset),
                               preprocessing=cache)

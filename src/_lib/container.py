
from dependency_injector import containers, providers

from src.config.logger import setup_logging
from src.domain.entities.Model_Settings import ModelSettings
from src.infrastructure.r_model.core.build_model import BuildModel
from src.infrastructure.r_model.core.build_tokenizer import BuildTokenizer


def _load_model(settings: ModelSettings) -> BuildModel:
    return BuildModel(settings).load()


def _load_tokenizer(settings: ModelSettings) -> BuildTokenizer:
    return BuildTokenizer(settings, padding_side="left").load()


class Container(containers.DeclarativeContainer):
    """Dependency injection container for application services and configuration."""

    logger = providers.Singleton(setup_logging)

    model_settings = providers.Singleton(ModelSettings)

    model = providers.Singleton(_load_model, settings=model_settings)

    tokenizer = providers.Singleton(_load_tokenizer, settings=model_settings)


class ContainerSingleton:
    """Singleton wrapper for the DI container"""
    _instance: Container | None = None

    @classmethod
    def get_instance(cls) -> Container:
        """Get the singleton container instance"""
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance


def get_container() -> Container:
    """Get the global container instance"""
    return ContainerSingleton.get_instance()

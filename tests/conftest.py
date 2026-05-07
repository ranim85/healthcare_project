import pytest

from src.utils.sklearn import configure_sklearn_output


@pytest.fixture(scope="session", autouse=True)
def configure_sklearn():
    """Garantit que tous les tests utilisent la sortie Pandas de sklearn."""
    configure_sklearn_output()

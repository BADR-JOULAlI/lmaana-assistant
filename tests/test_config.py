from sawtma.config import settings


def test_default_settings():
    assert settings.app_name == "SawtMA"
    assert settings.environment == "development"


from app.config import load_settings


def test_generic_llm_environment_takes_priority(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_PROTOCOL", "chat_completions")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_MODEL", "generic-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MAX_TOOL_ROUNDS", "4")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "legacy-key")

    settings = load_settings()

    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_protocol == "chat_completions"
    assert settings.llm_api_key == "generic-key"
    assert settings.llm_model == "generic-model"
    assert settings.llm_base_url == "https://llm.example/v1"
    assert settings.llm_max_tool_rounds == 4


def test_legacy_deepseek_environment_remains_compatible(monkeypatch) -> None:
    for name in (
        "LLM_PROVIDER",
        "LLM_PROTOCOL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_MAX_TOOL_ROUNDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "legacy-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "legacy-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://legacy.example")
    monkeypatch.setenv("DEEPSEEK_MAX_TOOL_ROUNDS", "3")

    settings = load_settings()

    assert settings.llm_provider == "deepseek"
    assert settings.llm_protocol == "responses"
    assert settings.llm_api_key == "legacy-key"
    assert settings.llm_model == "legacy-model"
    assert settings.llm_base_url == "https://legacy.example"
    assert settings.llm_max_tool_rounds == 3

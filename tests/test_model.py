import model


def test_extract_text_bloom_style():
    body = [{"generated_text": "Hello world"}]
    assert model._extract_text(body) == "Hello world"


def test_extract_text_flan_summary():
    body = [{"summary_text": "Short summary"}]
    assert model._extract_text(body) == "Short summary"


def test_extract_text_error_dict():
    body = {"error": "Model is loading"}
    assert "ERROR" in model._extract_text(body)


def test_should_retry_503():
    class R:
        status_code = 503

    assert model._should_retry(R(), {}) is True


def test_should_retry_loading_message():
    class R:
        status_code = 200

    assert model._should_retry(
        R(), {"error": "Model openai-community/gpt2 is currently loading"}
    ) is True

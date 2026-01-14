from fastapi.testclient import TestClient

from pocket_tts import main as main_module
from pocket_tts.main import web_app
from pocket_tts.presets import list_presets
from pocket_tts.utils.utils import PREDEFINED_VOICES


def test_presets_endpoint():
    client = TestClient(web_app)
    response = client.get("/presets")
    assert response.status_code == 200
    data = response.json()
    names = {preset["name"] for preset in data["presets"]}
    assert set(list_presets()) == names


def test_voices_endpoint():
    client = TestClient(web_app)
    response = client.get("/voices")
    assert response.status_code == 200
    data = response.json()
    names = {voice["name"] for voice in data["voices"]}
    assert names == set(PREDEFINED_VOICES)


def test_metadata_endpoint():
    class _DummyModel:
        sample_rate = 24000
        device = "cpu"
        has_voice_cloning = True

    main_module.tts_model = _DummyModel()
    client = TestClient(web_app)
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["sample_rate"] == 24000
    assert data["device"] == "cpu"
    assert data["has_voice_cloning"] is True

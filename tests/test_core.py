"""Tests for realtime-super-res."""
import pytest
from realtime_super_res import Pipeline, Config

class TestPipeline:
    def test_init(self):
        config = Config(device="cpu")
        pipeline = Pipeline(config=config, device="cpu")
        assert pipeline.config.device == "cpu"

    def test_run(self):
        config = Config(device="cpu")
        pipeline = Pipeline(config=config, device="cpu")
        result = pipeline.run("test_input")
        assert result["status"] == "completed"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

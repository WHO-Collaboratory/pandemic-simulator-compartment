"""Integration tests for full simulation workflows"""
import pytest
import json
from pathlib import Path
from datetime import datetime

from compartment.models.abc_jax_model.main import load_config, process_config, run_simulation


class TestABCIntegration:
    """Integration tests for ABC model end-to-end"""
    
    def test_load_example_config(self, load_example_config):
        """Test loading example config file"""
        config = load_example_config("abc_jax_model")
        
        assert config is not None
        assert "Disease" in config
        assert "case_file" in config
    
    def test_process_config(self, load_example_config):
        """Test config processing"""
        raw_config = load_example_config("abc_jax_model")
        
        processed = process_config(raw_config)
        
        assert "compartment_list" in processed
        assert "initial_population" in processed
        assert "start_date" in processed
        assert isinstance(processed["start_date"], datetime)
    
    def test_full_simulation_pipeline(self, load_example_config, tmp_path):
        """Test complete simulation from config to output"""
        config = load_example_config("abc_jax_model")
        
        # Process config
        processed_config = process_config(config)
        processed_config["time_steps"] = 10  # Short run for testing
        
        # Run simulation
        result = run_simulation(processed_config)
        
        assert result is not None
        assert "compartment_data" in result
        assert "metadata" in result
        
        # Verify output can be serialized to JSON
        output_file = tmp_path / "test_output.json"
        with open(output_file, "w") as f:
            json.dump(result, f, default=str)
        
        assert output_file.exists()
        assert output_file.stat().st_size > 0
    
    def test_simulation_with_interventions(self, abc_config):
        """Test simulation with interventions applied"""
        # Add an intervention
        abc_config["interventions"] = [{
            "start_date": datetime(2024, 1, 10),
            "end_date": datetime(2024, 1, 20),
            "intervention_type": "transmission_reduction",
            "value": 0.5,
            "admin_zones": "all"
        }]
        abc_config["time_steps"] = 30
        
        from compartment.models.abc_jax_model.model import ABCJaxModel
        model = ABCJaxModel(abc_config)
        result = model.simulate()
        
        assert result is not None
        assert "compartment_data" in result


class TestCovidIntegration:
    """Integration tests for COVID model"""
    
    def test_load_example_config(self, load_example_config):
        """Test loading COVID example config"""
        try:
            config = load_example_config("covid_jax_model")
            assert config is not None
        except:
            pytest.skip("COVID example config not available")
    
    def test_covid_simulation_runs(self, covid_config):
        """Test that COVID simulation completes"""
        covid_config["time_steps"] = 10
        
        from compartment.models.covid_jax_model.model import CovidJaxModel
        model = CovidJaxModel(covid_config)
        result = model.simulate()
        
        assert result is not None


class TestMpoxIntegration:
    """Integration tests for Mpox model"""
    
    def test_mpox_simulation_runs(self, mpox_config):
        """Test that Mpox simulation completes"""
        mpox_config["time_steps"] = 10
        
        from compartment.models.mpox_jax_model.model import MpoxJaxModel
        model = MpoxJaxModel(mpox_config)
        result = model.simulate()
        
        assert result is not None


class TestDengueIntegration:
    """Integration tests for Dengue model"""
    
    def test_dengue_simulation_runs(self, dengue_config):
        """Test that Dengue simulation completes"""
        dengue_config["time_steps"] = 10
        
        from compartment.models.dengue_jax_model.model import DengueJaxModel
        model = DengueJaxModel(dengue_config)
        result = model.simulate()
        
        assert result is not None


class TestCrossModelIntegration:
    """Integration tests across multiple models"""
    
    def test_all_models_produce_valid_output(self, abc_config, covid_config, mpox_config, dengue_config):
        """Test that all models produce valid, serializable output"""
        from compartment.models.abc_jax_model.model import ABCJaxModel
        from compartment.models.covid_jax_model.model import CovidJaxModel
        from compartment.models.mpox_jax_model.model import MpoxJaxModel
        from compartment.models.dengue_jax_model.model import DengueJaxModel
        
        configs = [
            (ABCJaxModel, abc_config),
            (CovidJaxModel, covid_config),
            (MpoxJaxModel, mpox_config),
            (DengueJaxModel, dengue_config)
        ]
        
        for ModelClass, config in configs:
            config["time_steps"] = 5
            model = ModelClass(config)
            result = model.simulate()
            
            # Should be JSON serializable
            json_str = json.dumps(result, default=str)
            assert len(json_str) > 0
    
    def test_consistent_output_format(self, abc_config, covid_config):
        """Test that different models produce similarly structured output"""
        from compartment.models.abc_jax_model.model import ABCJaxModel
        from compartment.models.covid_jax_model.model import CovidJaxModel
        
        abc_config["time_steps"] = 5
        covid_config["time_steps"] = 5
        
        abc_model = ABCJaxModel(abc_config)
        covid_model = CovidJaxModel(covid_config)
        
        abc_result = abc_model.simulate()
        covid_result = covid_model.simulate()
        
        # Both should have same top-level keys
        assert "compartment_data" in abc_result
        assert "compartment_data" in covid_result
        assert "metadata" in abc_result
        assert "metadata" in covid_result

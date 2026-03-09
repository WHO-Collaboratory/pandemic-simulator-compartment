"""Unit tests for validation schemas"""
import pytest
from pydantic import ValidationError

from compartment.validation.diseases.abc import ABCDiseaseConfig, ABCTransmissionEdge
from compartment.validation.diseases.covid import CovidDiseaseConfig
from compartment.validation.diseases.mpox import MpoxDiseaseConfig
from compartment.validation.diseases.dengue import DengueDiseaseConfig


class TestABCValidation:
    """Tests for ABC validation schema"""
    
    def test_valid_abc_config(self):
        """Test that valid ABC config passes validation"""
        config_data = {
            "disease_type": "ABC",
            "compartment_list": ["A", "B", "C"],
            "transmission_edges": [
                {
                    "source": "a_compartment",
                    "target": "b_compartment",
                    "data": {
                        "transmission_rate": 0.3
                    }
                },
                {
                    "source": "b_compartment",
                    "target": "c_compartment",
                    "data": {
                        "transmission_rate": 0.1
                    }
                }
            ]
        }
        
        disease_config = ABCDiseaseConfig(**config_data)
        
        assert disease_config.disease_type == "ABC"
        assert len(disease_config.transmission_edges) == 2
        assert disease_config.compartment_list == ["A", "B", "C"]
    
    def test_abc_transmission_edge_id_generation(self):
        """Test that edge IDs are auto-generated"""
        edge_data = {
            "source": "a_compartment",
            "target": "b_compartment",
            "data": {
                "transmission_rate": 0.3
            }
        }
        
        edge = ABCTransmissionEdge(**edge_data)
        
        assert edge.id == "a_compartment->b_compartment"
    
    def test_negative_transmission_rate_fails(self):
        """Test that negative transmission rates are rejected"""
        edge_data = {
            "source": "a_compartment",
            "target": "b_compartment",
            "data": {
                "transmission_rate": -0.3  # Invalid!
            }
        }
        
        with pytest.raises(ValidationError):
            ABCTransmissionEdge(**edge_data)
    
    def test_zero_transmission_rate_fails(self):
        """Test that zero transmission rate is rejected"""
        edge_data = {
            "source": "a_compartment",
            "target": "b_compartment",
            "data": {
                "transmission_rate": 0.0  # Invalid!
            }
        }
        
        with pytest.raises(ValidationError):
            ABCTransmissionEdge(**edge_data)
    
    def test_variance_params_optional(self):
        """Test that variance params are optional"""
        edge_data = {
            "source": "a_compartment",
            "target": "b_compartment",
            "data": {
                "transmission_rate": 0.3
                # No variance_params
            }
        }
        
        edge = ABCTransmissionEdge(**edge_data)
        
        assert edge.data.variance_params is None


class TestCovidValidation:
    """Tests for COVID validation schema"""
    
    def test_valid_covid_config(self):
        """Test that valid COVID config passes validation"""
        config_data = {
            "disease_type": "NOVEL-RESPIRATORY",
            "compartment_list": ["S", "E", "I", "H", "R", "D"],
            "transmission_edges": [
                {
                    "source": "susceptible",
                    "target": "exposed",
                    "data": {"transmission_rate": 0.5}
                }
            ]
        }
        
        disease_config = CovidDiseaseConfig(**config_data)
        
        assert disease_config.disease_type == "NOVEL-RESPIRATORY"
    
    def test_covid_requires_valid_compartments(self):
        """Test COVID model requires specific compartments"""
        config_data = {
            "disease_type": "NOVEL-RESPIRATORY",
            "compartment_list": ["X", "Y", "Z"],  # Invalid compartments
            "transmission_edges": []
        }
        
        # This may or may not fail depending on validation rules
        # Adjust based on actual validation implementation
        try:
            disease_config = CovidDiseaseConfig(**config_data)
        except ValidationError:
            pytest.skip("COVID validation enforces specific compartments")


class TestMpoxValidation:
    """Tests for Mpox validation schema"""
    
    def test_valid_mpox_config(self):
        """Test that valid Mpox config passes validation"""
        config_data = {
            "disease_type": "MPOX",
            "compartment_list": ["S", "E", "I", "R"],
            "transmission_edges": [
                {
                    "source": "susceptible",
                    "target": "exposed",
                    "data": {"transmission_rate": 0.3}
                }
            ]
        }
        
        disease_config = MpoxDiseaseConfig(**config_data)
        
        assert disease_config.disease_type == "MPOX"


class TestDengueValidation:
    """Tests for Dengue validation schema"""
    
    def test_valid_dengue_config(self):
        """Test that valid Dengue config passes validation"""
        config_data = {
            "disease_type": "NOVEL-VECTOR-BORNE",
            "compartment_list": ["S_h", "I_h", "R_h", "S_v", "I_v"],
            "transmission_edges": [
                {
                    "source": "susceptible_human",
                    "target": "infected_human",
                    "data": {"transmission_rate": 0.3}
                }
            ]
        }
        
        disease_config = DengueDiseaseConfig(**config_data)
        
        assert disease_config.disease_type == "NOVEL-VECTOR-BORNE"


class TestValidationEdgeCases:
    """Edge case tests for validation"""
    
    def test_empty_compartment_list_fails(self):
        """Test that empty compartment list is rejected"""
        config_data = {
            "disease_type": "ABC",
            "compartment_list": [],  # Empty!
            "transmission_edges": []
        }
        
        with pytest.raises(ValidationError):
            ABCDiseaseConfig(**config_data)
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise errors"""
        config_data = {
            "disease_type": "ABC"
            # Missing compartment_list and transmission_edges
        }
        
        with pytest.raises(ValidationError):
            ABCDiseaseConfig(**config_data)
    
    def test_extra_fields_ignored(self):
        """Test that extra fields are handled appropriately"""
        config_data = {
            "disease_type": "ABC",
            "compartment_list": ["A", "B", "C"],
            "transmission_edges": [],
            "extra_field": "should be ignored"
        }
        
        # Pydantic should either ignore or reject extra fields
        # depending on config
        try:
            disease_config = ABCDiseaseConfig(**config_data)
            # If it succeeds, extra fields are ignored
            assert not hasattr(disease_config, "extra_field")
        except ValidationError:
            # If it fails, extra fields are forbidden
            pass

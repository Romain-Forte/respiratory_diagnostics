
from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum
from datetime import datetime

class HematologicDiseaseTypeEnum(str, Enum):
    aml = "aml"
    myelome = "myelome"
    all = "all"
    cll = "cll"
    lymphomeHodgkin = "lymphomeHodgkin"
    lymphomeNonHodgkin = "lymphomeNonHodgkin"
    mds = "mds"
    other = "other"

class GraftTypeEnum(str, Enum):
    no_graft = "no_graft"
    autograft = "autograft"
    allograft = "allograft"
    
class Symptoms(BaseModel):
    sex: str
    age: Optional[int]
    time_hospital_to_icu: Optional[int] = Field(None, ge=0)
    time_symptoms_to_icu: Optional[int] = Field(None, ge=0)
    time_diagnosis_to_icu: Optional[int] = Field(None, ge=0)
    hematologic_diseases: list[HematologicDiseaseTypeEnum] = Field(default_factory=list)
    graft_type: GraftTypeEnum
    graft_rejection: bool
    systemic_disease: bool
    organ_transplant: bool
    solid_tumor: bool
    immunosuppression: bool
    chemotherapy: bool
    ibrutinib_fludarabine_methotrexate: bool
    targeted_therapy: bool
    immunotherapy: bool
    car_t_cells: bool
    steroids: bool
    pneumocystosis: bool
    antifungal_prophylaxis: bool
    bacterial_prophylaxis: bool
    viral_prophylaxis: bool
    influenza_vaccine: bool
    covid19_vaccine: bool
    other_vaccines: bool
    sofa_score: int = Field(..., ge=0, le=24)
    glasgow_score: Optional[int] = Field(None, ge=3, le=15)
    respiratory_rate: int = Field(..., gt=0)
    oxygen_saturation: float = Field(..., ge=0.0, le=100.0)
    temperature: float
    pao2_fio2_ratio: float = Field(..., gt=0)
    septic_shock: bool
    neutropenia: bool
    alveolar_lung_involvement: bool
    interstitial_lung_involvement: bool
    quadrant_count: int = Field(..., ge=0, le=4)
    fibrosis: bool
    nodules: bool
    pneumothorax: bool
    cardiomegaly: bool
    lung_opacity: bool
    septal_lines: bool
    pleural_effusion: bool


class PredictRequest(BaseModel):
    request_id: str = Field(..., pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    service_id: str = Field(..., pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    timestamp: datetime
    symptoms: Symptoms
{
    "request_id": "1a0b5056-861a-4af1-8315-65e4c58770f0", # str, identifiant UUID de la requête
    "service_id": "14bb75a7-32b6-40d2-964d-ca2265a2f102", # str, identifiant UUID du service hospitalier
    "timestamp": "2026-01-25T14:32:00Z", # str, format ISO 8601
    "symptoms": {
        "sex": "man", # Options: "man", "woman", "other"
        "age": 53, # int, en années
        "time_hospital_to_icu": 2, # int, en jours
        "time_symptoms_to_icu": 5, # int, en jours
        "time_diagnosis_to_icu": "", # int, en jours
        "hematologic_diseases": [
            { "type": "aml", "controlled": False }, # types possibles : 'aml', 'myelome','all','cll', 'lymphomeHodgkin', 'lymphomeNonHodgkin', 'mds', 'other'
            { "type": "myelome", "controlled": True }
        ],
        "graft_type": "no_graft", # Options: "no_graft", "autograft", "allograft"
        "graft_rejection": False, # bool
        "systemic_disease": False, # bool
        "organ_transplant": False, # bool
        "solid_tumor": False, # bool
        "immunosuppression": False, # bool
        "chemotherapy": False, # bool
        "ibrutinib_fludarabine_methotrexate": False, # bool
        "targeted_therapy": False, # bool
        "immunotherapy": False, # bool
        "car_t_cells": False, # bool
        "steroids": False, # bool
        "pneumocystosis": False, # bool
        "antifungal_prophylaxis": False, # bool
        "bacterial_prophylaxis": False, # bool
        "viral_prophylaxis": False, # bool
        "influenza_vaccine": False, # bool
        "covid19_vaccine": False, # bool
        "other_vaccines": False, # bool
        "sofa_score": 3, # int, compris entre 0 et 24
        "glasgow_score": "", # int, compris entre 3 et 15
        "respiratory_rate": 12, # int, supérieur à 0
        "oxygen_saturation": 99.2, # float, en pourcentage
        "temperature": 38.1, # float, en °C
        "pao2_fio2_ratio": 250.0, # float, en mmHg
        "septic_shock": False, # bool
        "neutropenia": False, # bool
        "alveolar_lung_involvement": False, # bool
        "interstitial_lung_involvement": False, # bool
        "quadrant_count": 1, # int, entre 0 et 4
        "fibrosis": False, # bool
        "nodules": False, # bool
        "pneumothorax": False, # bool
        "cardiomegaly": False, # bool
        "lung_opacity": False, # bool
        "septal_lines": False, # bool
        "pleural_effusion": False, # bool
    },
    "missing_features": ["time_diagnosis_to_icu", "glasgow_score"] # list of str, noms des features non renseignées (hors features booléennes)
}
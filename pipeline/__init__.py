"""
Twinship Anomaly Detection Pipeline package.

Stages
──────
01 – EDA               (pipeline.stage_01_eda)
02 – Data Cleaning     (pipeline.stage_02_cleaning)
03 – Clustering        (pipeline.stage_03_clustering)
04 – Anomaly Detection (pipeline.stage_04_anomaly_detection)
05 – Classification    (pipeline.stage_05_classification)
06 – Saving            (pipeline.stage_06_saving)

Orchestrator
────────────
    from pipeline.run_pipeline import run_pipeline
    results = run_pipeline()
"""
from .run_pipeline import run_pipeline

__all__ = ['run_pipeline']

# Protean Geo-Analytics Platform
### A Hybrid Neuro-Symbolic Approach to Seismic Risk Assessment

## From Possibility to Performance

The Protean Platform is a seismic interpretation tool designed to assess subsurface risk. Instead of relying solely on manual fault detection, this system utilizes an AI co-pilot to quantify risk. The platform combines Generative AI, Graph Theory, and Large Language Models (LLMs) to convert uncertainty into actionable insights.

## Project Overview

In Geotechnical Engineering, traditional seismic data interpretation is often manual, subjective, and slow. Standard maps typically identify where a fault is located but fail to predict if it presents a leakage risk.

The Protean Solution automates this process using three distinct AI agents that handle visual recognition, mathematical analysis, and textual reporting.

## System Architecture

The platform consists of three integrated modules:

| Module | Name | Role | Tech Stack | Function |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1** | **REE** | Visual Extraction | Attention U-Net (PyTorch) | Generates detailed fault maps, effectively handling noisy data. |
| **Module 2** | **GPP** | Structural Analysis | Graph Attention Network (PyG) | Calculates connectivity to determine leakage risks. |
| **Module 3** | **MAIA** | Reporting | Gemini API (LLM) | Converts complex mathematical metrics into readable reports. |

## Methodology

The system processes raw seismic data into insights through the following pipeline:

### 1. The Reality Ensemble Engine (REE)
REE functions as the vision expert, designed to identify features in noisy data.

* **Input:** A slice of 2D Seismic Amplitude (derived from a 3D volume).
* **Model:** A modified U-Net utilizing Attention Gates.
* **Mechanism:** Since seismic data contains significant background noise (approximately 90%), standard CNNs often struggle. Attention Gates force the model to focus specifically on fault features.
* **Output:** A binary map indicating fault locations.

### 2. The Graph Property Predictor (GPP)
GPP analyzes the connectivity of the identified faults.

* **Transformation:** The system converts fault lines from the visual map into a network graph.
* **Nodes & Edges:** Fault lines serve as nodes, while their intersections/connections serve as edges.
* **Model:** A Graph Attention Network (GAT). This model weighs neighbor importance to identify flow paths.
* **Prediction:** It classifies the network as High Risk (leakage likely) or Low Risk (sealing) based on vertical connectivity.

### 3. The Multimodal AI Analyst (MAIA)
MAIA synthesizes the data into a natural language report.

* **Integration:** Risk scores and connectivity metrics are aggregated into a JSON file.
* **Reasoning:** The system uses Retrieval Augmented Generation (RAG) powered by the Gemini 1.5 Pro API.
* **Prompting:** The LLM acts as a Senior Geophysicist, analyzing connectivity numbers to assess leakage risks.
* **Output:** A comprehensive written report with actionable advice.

## Dataset

This project utilizes the industrial-grade Gullfaks Field Dataset from the North Sea.

| Data Type | Source Format | Role | Pre-processing |
| :--- | :--- | :--- | :--- |
| **3D Seismic Volume** | .segy | Primary Input (X) | Sliced into 128x128 2D patches for training. |
| **Fault Sticks** | ASCII | Target Labels (Y) | Vector lines converted to pixel masks. |
| **Well Logs** | .las | Lithology Stats | K-Means clustering used to determine rock types (Sand vs. Shale). |

## Installation and Usage

### Prerequisites
* Python 3.8 or newer
* NVIDIA GPU (Recommended for training)

### Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Lakshya44444/protean-geo-platform.git](https://github.com/Lakshya44444/protean-geo-platform.git)
    cd protean-geo-platform
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    Launch the Streamlit dashboard:
    ```bash
    streamlit run app.py
    ```

## Roadmap

* [x] **Week 1:** Data pipeline implementation (SEGY slicing).
* [x] **Week 1:** Graph Construction algorithm development.
* [ ] **Week 2:** Training Attention U-Net on Gullfaks data.
* [ ] **Week 3:** Integration of Gemini API for reasoning and reporting.
* [ ] **Week 4:** Deployment to Streamlit Cloud.

## References

* **Liu et al. (2021):** Attention-Based 3-D Seismic Fault Segmentation.
* **Li et al. (2024):** Graph Network Surrogate Model for Subsurface Flow.
* **Zhang et al. (2024):** When Geoscience Meets Generative AI.

## Author

**Lakshya Gupta**
Sophomore, Geological Technology, IIT Roorkee

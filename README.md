Protean Geo-Analytics Platform 🌍🤖

A Hybrid Neuro-Symbolic Approach to Seismic Risk Assessment

"From Possibility to Performance"

The Protean Platform represents a paradigm shift in subsurface modeling, evolving from static, deterministic interpretation to a dynamic ecosystem of probabilistic risk assessment. Acting as an AI-powered co-pilot for geologists, it synergizes the creative power of Generative AI, the structural reasoning of Graph Theory, and the interpretive capabilities of Large Language Models to quantify uncertainty in high-stakes geotechnical environments.

🚀 Project Overview

In the domain of Geotechnical Engineering, the conventional workflow for seismic interpretation remains heavily reliant on manual, subjective analysis. This traditional approach often results in static geological models that tell us where a fault might be located geographically, but fail to quantify how risky that structure is in terms of fluid migration and seal integrity.

The Protean Solution addresses this critical gap by automating the entire interpretation-to-analysis pipeline. It deploys three specialized, interoperable AI agents that mirror the cognitive workflow of a human expert team:



### 🧠 System Architecture: The "Triad" Approach

| Module Name | The Metaphor | Tech Stack | Core Function |
| :--- | :--- | :--- | :--- |
| **Module 1: REE** | 🎨 **The Artist** | **Attention U-Net** (PyTorch) | **Visualizes:** Generates high-fidelity fault maps from noisy seismic data. |
| **Module 2: GPP** | 👷 **The Engineer** | **Graph Attention Network** (PyG) | **Calculates:** Simulates physical connectivity to predict leakage risk. |
| **Module 3: MAIA** | 📊 **The Analyst** | **Gemini API** (LLM) | **Explains:** Interprets risk metrics into natural language reports. |

🛠️ Methodology & Technical Details

Our pipeline transforms raw, unstructured seismic data into actionable risk insights through a rigorous, multi-stage process designed to handle real-world industrial complexity:

1. The Reality Ensemble Engine (REE)

The REE serves as the vision core of the platform. It addresses the challenge of identifying subtle geological features within inherently noisy seismic volumes.

Input: 2D Seismic Amplitude Slice (extracted from 3D SEGY Volume).

Model: A modified U-Net architecture enhanced with Attention Gates.

Why Attention? Seismic data is typically >90% background noise and geological strata that are irrelevant to fault detection. Standard CNNs often "get distracted" by these textures. 

 Attention Gates introduce a learnable weighting mechanism that forces the model to suppress irrelevant regions and focus gradients exclusively on the structural discontinuities (the signal), significantly improving segmentation accuracy in low-quality data.

Output: A precise binary probability mask delineating the fault structure.

2. The Graph Property Predictor (GPP)

The GPP is the physics-aware engine that moves beyond simple image recognition to understand topological relationships.

Transformation: We employ a custom Image-to-Graph algorithm to skeletonize the predicted fault mask into a formal topological graph structure.

Nodes: represent discrete fault segments and critical intersections.

Edges: represent the physical connectivity and transmissibility between these segments.

Model: A Graph Attention Network (GAT) that processes this graph structure. Unlike standard Graph Convolutional Networks (GCNs), the GAT applies an attention mechanism to weigh the importance of neighboring nodes, allowing it to identify critical flow pathways. It classifies the entire graph structure as "Leaking" (High Risk) or "Sealing" (Low Risk) based on vertical connectivity rules that simulate fluid migration potential.

3. The Multimodal AI Analyst (MAIA)

MAIA acts as the reasoning layer, translating abstract mathematical outputs into human-readable intelligence.

Integration: The probabilistic risk scores and connectivity metrics from the GPP are aggregated and converted into a structured JSON payload.

Reasoning: We utilize a Retrieval Augmented Generation (RAG) framework powered by the Gemini 1.5 Pro API.

System Prompt: The model is primed with a specialized persona: "Act as a Senior Geophysicist. Analyze the following connectivity metrics and provide a comprehensive risk assessment, highlighting potential leakage pathways between the reservoir and overburden..."

Output: A detailed text-based strategic report that provides context, interpretation, and actionable recommendations for the user.

📂 Data Ecosystem

To ensure the system is robust and applicable to real-world scenarios, we utilize the industrial-grade Gullfaks Field Dataset from the North Sea, a standard benchmark for complex structural geology.

Data Type

Source Format

Role in Project

Key Transformation

3D Seismic Volume

.segy

Primary Input ($X$)

The raw 3D volume is sliced into 2D patches ($128 \times 128$) to create a manageable training set for the vision model.

Fault Sticks

ASCII

Ground Truth Labels ($Y$)

Vector fault interpretations are rasterized into binary masks to serve as pixel-perfect training targets.

Well Logs

.las

Lithology Features

Gamma Ray and Porosity logs are processed using K-Means clustering to classify rock types (Sand/Shale) for graph node attribution.

Access the Dataset Here

💻 Installation & Usage

Prerequisites

Python 3.8+

NVIDIA GPU (Recommended for training)

1. Clone the Repository

git clone [https://github.com/Lakshya44444/protean-geo-platform.git](https://github.com/Lakshya44444/protean-geo-platform.git)
cd protean-geo-platform


2. Install Dependencies

pip install -r requirements.txt


3. Run the Demo App

Launch the Streamlit dashboard to test the pipeline:

streamlit run app.py


🧪 Current Status (Roadmap)

[x] Week 1: Data Engineering pipeline (SEGY slicing) complete.

[x] Week 1: Graph Construction algorithm (Image-to-Graph) implemented.

[ ] Week 2: Training the Attention U-Net on Gullfaks data.

[ ] Week 3: Integrating the Gemini API for reasoning.

[ ] Week 4: Final Deployment on Streamlit Cloud.

📚 References

Liu et al. (2021): Attention-Based 3-D Seismic Fault Segmentation. (Theoretical Basis for Vision Model).

Li et al. (2024): Graph Network Surrogate Model for Subsurface Flow. (Theoretical Basis for Physics Model).

Zhang et al. (2024): When Geoscience Meets Generative AI. (Theoretical Basis for LLM Integration).I

👨‍💻 Author

**Lakshya Gupta

---
tags: [unsupervised, tabular, federated learning, anomaly detection]
dataset: [Health Insurance Claim Prediction]
framework: [scikit-learn, tensorflow, flwr]
---

# Federated-Health-Anomaly-Detection
*A Federated Learning Framework for Anomaly Detection on Heterogeneous Health Insurance Claims*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

---

### **Project Canvas**

| Feature | Details |
| :--- | :--- |
| 📖 **Description** | A privacy-preserving framework using Flower to train an Autoencoder for unsupervised anomaly detection on non-IID health insurance claims. |
| 💾 **Dataset** | **Health Insurance Claim Prediction** by Shivam Kumar Bhagat ([Source](https://github.com/bhagat-shivam/Health-Insurace-claim-prediction)) |
| 🛠️ **Frameworks** | `Flower`, `TensorFlow`, `Scikit-learn`, `Pandas`, `SHAP` |
| ⚖️ **License** | Code is licensed under **MIT**. See `LICENSE.txt`. <br> Data is licensed under a separate **MIT** license. See `Source Data/LICENSE-DATA.txt`. |
| ✍️ **Citation** | [Validi, Z., Albadvi, A., et al. (2025). *Applied Artificial Intelligence*.](#citation) |

---

## 🚀 Getting Started: Quickstart

Follow these steps to set up the environment and reproduce the results from the paper.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/zahravalidi/Federated-Health-Anomaly-Detection.git](https://github.com/zahravalidi/Federated-Health-Anomaly-Detection.git)
    cd Federated-Health-Anomaly-Detection
    ```

2.  **Create Environment & Install Dependencies**
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install the project and its dependencies
    pip install -e .
    ```

3.  **Run the Analysis Notebook**
    The main notebook contains the complete workflow to reproduce the paper's findings.
    ```bash
    # Launch Jupyter Lab
    jupyter lab

    # In Jupyter, navigate to the Notebooks/ directory and open the main analysis notebook.
    ```

---

## 📂 Repository Structure

The repository is organized to clearly separate code, data, notebooks, and configuration.

```
.
├── LICENSE.txt               # The license for this project's code (MIT)
├── pyproject.toml            # Python project configuration
├── README.md                 # This file
│
├── App/                      # Core federated learning application logic
│   ├── client_app.py
│   ├── server_app.py
│   └── task.py
│
├── Notebooks/                # Jupyter notebooks for analysis and visualization
│   └── main_analysis.ipynb
│
├── Src/                      # Python script versions of the notebooks
│   └── ...
│
├── Source Data/              # Raw data and its license
│   ├── health.csv
│   └── LICENSE-DATA.txt
│
└── Generated Artifacts/      # (Ignored by Git) For saved models (.pkl)
```

---

## 🔬 In-Depth Usage: Running the FL Simulation

To run the federated learning system with a live server and clients, you will need multiple terminals.

1.  **Start the FL Server:**
    In your first terminal, start the server. It will wait for clients to connect.
    ```bash
    python App/server_app.py
    ```

2.  **Start FL Clients:**
    In a second terminal (and third, etc.), start a client. Each client will connect to the server and begin training.
    ```bash
    python App/client_app.py
    ```

---

## ✍️ Authorship & Citation

* **Code Author:** Zahra Validi
* **Research Supervision:** Amir Albadvi, Elham Akhondzadeh Noughabi, Mitra Ghanbarzadeh

The source code was developed by Zahra Validi under the supervision and guidance of the co-authors.

<details>
<summary><strong>View BibTeX Citation</strong></summary>

```bibtex
@article{Validi2025,
  author    = {Validi, Zahra and Albadvi, Amir and Noughabi, Elham Akhondzadeh and Ghanbarzadeh, Mitra},
  title     = {A Federated Learning Framework for Anomaly Detection on Heterogeneous Health Insurance Claims},
  journal   = {Applied Artificial Intelligence},
  year      = {2025},
}
```
</details>

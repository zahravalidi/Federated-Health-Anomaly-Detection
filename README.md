---
tags: [unsupervised, tabular, federated learning, anomaly detection]
dataset: [Health Insurance Claim Prediction]
framework: [scikit-learn, tensorflow, flwr]
---

# Federated-Health-Anomaly-Detection
### A Federated Learning Framework for Anomaly Detection on Heterogeneous Health Insurance Claims

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

---

## 📖 Overview

This project implements and evaluates a privacy-preserving federated learning (FL) framework for unsupervised anomaly detection on a real-world, non-identically distributed (non-IID) health insurance claims dataset. The goal is to identify high-risk, anomalous claims without centralizing sensitive patient data.

To simulate a realistic, non-IID federated environment, the dataset is partitioned among 91 clients based on the "city" feature. A deep learning **Autoencoder** is trained to identify anomalous claims. The federated learning process is performed using the **FedProx** strategy over 20 communication rounds to mitigate the effects of statistical heterogeneity.

---

## 📂 Project Structure

The repository is organized to separate configuration, source code, notebooks, and data.

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
│   └── main_analysis.ipynb   # Main notebook to reproduce paper results
│
├── Src/                      # Python script versions of the notebooks
│   └── ...
│
├── Source Data/              # Raw data and its specific license
│   ├── health.csv
│   └── LICENSE-DATA.txt
│
└── Generated Artifacts/      # (Ignored by Git) For saved models (.pkl)


---

## 🚀 Setup and Installation

To get started with this project, follow the steps below.

### ### Prerequisites
- Python 3.9 or higher
- `pip` package manager
- A virtual environment tool like `venv`

### ### Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zahravalidi/Federated-Health-Anomaly-Detection.git](https://github.com/zahravalidi/Federated-Health-Anomaly-Detection.git)
    cd Federated-Health-Anomaly-Detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The required packages are defined in `pyproject.toml`. Install the project in editable mode:
    ```bash
    pip install -e .
    ```

---

## 📊 How to Run

### ### Reproduce Paper Results with Notebooks
The easiest way to explore the project and reproduce the figures and tables from the associated paper is by using the main Jupyter notebook.

1.  Navigate to the `Notebooks/` directory.
2.  Launch Jupyter Lab: `jupyter lab`
3.  Open `main_analysis.ipynb` and run the cells sequentially.

### ### Run the Federated Learning Simulation 🤖
To run the full federated system from the command line, you will need at least two separate terminal windows (with the virtual environment activated in each).

1.  **Start the FL Server:**
    In your first terminal, run the server application. The server will wait for clients to connect.
    ```bash
    python App/server_app.py
    ```

2.  **Start one or more FL Clients:**
    In a second terminal, run the client application to begin the training process. You can open additional terminals to run more clients simultaneously.
    ```bash
    python App/client_app.py
    ```

---

## 💿 Data

This project uses the publicly available **Health Insurance Claim Prediction** dataset, created by Shivam Kumar Bhagat.

-   **Source:** [GitHub Repository](https://github.com/bhagat-shivam/Health-Insurace-claim-prediction)
-   **License:** The dataset is provided under an MIT License. The full license text is available in the `Source Data/LICENSE-DATA.txt` file.

---

## ⚖️ Licensing

The source code for this project is original work and is licensed under the MIT License. See the `LICENSE.txt` file for more details.

---

## ✍️ Authorship and Acknowledgments

* **Code Author:** Zahra Validi
* **Research Supervision:** Amir Albadvi, Elham Akhondzadeh Noughabi, Mitra Ghanbarzadeh

This project is the result of a collaborative research effort. The source code was developed by Zahra Validi under the supervision and guidance of the co-authors of the associated academic paper.

---

## ## Citation

If you use this project in your research, please consider citing the original paper:

```bibtex
@article{Validi2025,
  author    = {Validi, Zahra and Albadvi, Amir and Noughabi, Elham Akhondzadeh and Ghanbarzadeh, Mitra},
  title     = {A Federated Learning Framework for Anomaly Detection on Heterogeneous Health Insurance Claims},
  journal   = {Applied Artificial Intelligence},
  year      = {2025},
}

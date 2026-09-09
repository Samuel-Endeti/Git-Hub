# Avian Intelligence Optimization Engine (AIWPSO & CNN)

An optimized deep learning image classification platform designed for high-accuracy performance in low-data availability environments, utilizing a Convolutional Neural Network (CNN) enhanced by an Adaptive Inertia Weight Particle Swarm Optimization (AIWPSO) algorithm.

Traditional machine learning pipelines struggle when training data is scarce. This project addresses data limitations by applying an optimized particle swarm algorithm to fine-tune network weights and feature selection, making rare bird species detection robust and reliable.

## 🚀 Live Application
Access the production-ready dashboard here: **[Open Live App](https://avian-intelligence-aiwpso.streamlit.app)**


## 🔑 Default Login Credentials
* **Standard User Account:** Username: `user` | Password: `user123`
* **Admin Account:** Username: `admin` | Password: `admin123`

## 🛠️ Key System Features
* **Optimized Engine:** Utilizes an evolutionary AIWPSO framework to enhance features and model weights under low data availability.
* **Deep Learning CNN:** Specialized deep learning architecture built for fine-grained image classification.
* **Smart Model Management:** Implemented on-the-fly binary stream reconstruction logic to seamlessly merge split high-capacity neural network chunks (`.h5`) directly within restricted cloud server environments.
* **Interactive Web App:** Multi-role operational workspaces built with Streamlit for a smooth user experience.

## 💻 Tech Stack
* **Language:** Python 3.10
* **Framework:** Streamlit Cloud Architecture
* **Core Deep Learning Framework:** TensorFlow, Keras
* **Algorithms & Optimization:** CNN, AIWPSO Metaheuristics
* **Data Processing & Utilities:** NumPy, Pillow, JSON

## ⚙️ How to Run Locally

Clone the repository to your machine:
```bash
git clone https://github.com
```

Navigate into the project directory:
```bash
cd avian-intelligence-aiwpso
```

Install the required software dependencies:
```bash
pip install -r requirements.txt
```

Launch the Streamlit web dashboard interface:
```bash
streamlit run app.py
```

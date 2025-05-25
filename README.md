# 🧠 Multi-Task NLP using T5 Transformer

This project leverages the power of a **masked T5 transformer model** to perform **Sentiment Analysis**, **Emotion Detection**, and **Text Summarization** on website comments. Each NLP task is handled in a separate pipeline, trained with custom loss functions and deployed using **Flask**. The training and experimentation were carried out using **Kaggle's GPU backend**.

---

## 🚀 Key Features

* ✅ Multi-task NLP with a unified T5 architecture
* 📊 Handles Sentiment, Emotion, and Summarization tasks separately
* 🧪 Custom loss functions for each NLP task
* 💻 Deployed via Flask with a simple frontend
* ⚙️ Trained using GPU on Kaggle

---

## 📁 Folder Structure

```
.
├── .ipynb_checkpoints/        # Notebook autosave checkpoints
├── data/                      # Training and evaluation datasets
├── model_notebooks/           # Notebooks for development & experimentation
├── multi_task_T5_tf/          # Main T5 model training scripts and pipeline
├── results/                   # Evaluation metrics, logs, and output files
├── static/css/                # Static CSS files for the Flask web UI
├── templates/                 # HTML templates for Flask
├── app.py                     # Main Flask backend to serve the model
├── checkpoint/                # Model checkpoints for loading weights
├── comments.json              # Input comments for testing inference
├── full_req.txt               # Full package list (from `pip freeze`)
├── requirements.txt           # Minimal required packages
├── Harshit Gupta Hack It Out.pdf # Project Report / Documentation
├── Procfile                   # Deployment profile (for platforms like Heroku)
├── Untitled.ipynb             # Miscellaneous notebook
├── .gitattributes             # Git LFS tracking
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git git@github.com:harshitgupta04022004/sentiment_pred_model.git
cd nlp-t5-multitask
```

### 2. Create & Activate Virtual Environment (Optional but Recommended)

```bash
conda create --name t5-nlp-env python=3.9
conda activate t5-nlp-env
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

To match the exact environment used in development:

```bash
pip install -r full_req.txt
```

---

## 🚦 Run the Flask App

```bash
python app.py
```

Then open your browser and go to:
**[http://localhost:5000/](http://localhost:8888/)**

---

## 🧪 How It Works

* `app.py` loads the appropriate pipeline (sentiment, emotion, or summary) based on user input.
* Each pipeline under `multi_task_T5_tf/` preprocesses text, passes it through the model, and decodes predictions.
* The HTML/CSS frontend is located in `templates/` and `static/css/`.

---


## 🧑‍💻 Author

**Harshit Gupta**
Feel free to raise an issue, star the repo, or contribute!

---

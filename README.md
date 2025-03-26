# Fake News Detection

## Setup Instructions

### 1. Create and Activate a Virtual Environment
```sh
python -m venv venv
```
- **Windows:**
  ```sh
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```sh
  source venv/bin/activate
  ```

### 2. Install Required Modules
```sh
pip install flask flask-cors pandas numpy scikit-learn nltk
```

### 3. Download NLTK Data
```sh
python -m nltk.downloader stopwords punkt
```

### 4. Run the Application
```sh
python app.py
python detector.py
```

### 5. Open the HTML File
- Open the frontend HTML file in your browser.

### 6. Test the Fake News Detection
- Copy a news article from `fake_news_temp.csv`.
- Paste it into the website.
- Click "Analyze" to check if the news is fake or real.

---
### Notes:
- Make sure `app.py` and `detector.py` are running before testing.
- Ensure you have `fake_news_temp.csv` in the project directory.
- If you encounter issues, check that all dependencies are installed correctly.



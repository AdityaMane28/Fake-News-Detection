# Fake News Detection

A Jupyter Notebook project that classifies news articles as **fake** or **real** using text preprocessing, TF-IDF features, and a Multinomial Naive Bayes model. The notebook also explores the dataset and reports classification metrics and visualizations.

## Project contents

```text
.
├── Fake_News_Detection.ipynb
├── README.md
└── requirements.txt
```

The notebook expects two CSV files that are not included in this repository:

- `Fake.csv` — articles labeled fake
- `True.csv` — articles labeled real

Each file should include at least `title`, `text`, and `subject` columns. The notebook reads the files with Latin-1 encoding. Please obtain the data from a source you are authorized to use and follow its license and attribution requirements.

## Run in Google Colab

1. Upload `Fake_News_Detection.ipynb`, `Fake.csv`, and `True.csv` to a Colab session.
2. Confirm that the CSV files are available at `/content/Fake.csv` and `/content/True.csv`, which are the paths used in the notebook.
3. Install the dependencies in a notebook cell:

   ```python
   %pip install -r requirements.txt
   ```

   If needed, upload `requirements.txt` to the session first, or install the listed packages directly with `%pip install numpy pandas matplotlib seaborn nltk wordcloud scikit-learn tqdm`.
4. Open the notebook and choose **Runtime → Run all**. The notebook downloads the NLTK English stopwords resource when it runs.

For local Jupyter use, install the requirements in your Python environment and update the two `pd.read_csv(...)` paths near the start of the notebook to point to your local copies of the CSV files.

## Workflow

1. Load and label the fake and real article datasets (`0 = fake`, `1 = real`), then combine and shuffle them.
2. Combine each article's title and body, lowercase the text, remove URLs and non-letter characters, remove English stopwords, and discard very short tokens.
3. Convert cleaned text into TF-IDF features with up to 10,000 features, unigrams and bigrams, `min_df=3`, `max_df=0.95`, and sublinear term frequency.
4. Split the data into training and test sets (80/20, random seed 42) and train `MultinomialNB` with `alpha=0.1`.
5. Report accuracy, a classification report, stratified 3-fold cross-validation accuracy, a confusion matrix, and ROC/AUC results. Plot the most informative terms for each class.
6. Save the trained classifier as `nb_model.pkl`, the TF-IDF vectorizer as `tfidf.pkl`, and the plots as `confusion_matrix.png`, `roc_curve.png`, and `top_keywords.png` in the notebook's working directory.
7. Use the notebook's `predict_news(title, text="")` function to classify a new article after training.

## Important evaluation note

The current notebook fits TF-IDF on the complete dataset before the train/test split and cross-validation. This lets information about the full corpus vocabulary reach the evaluation data, so reported scores can be somewhat optimistic. For a stricter evaluation, put text vectorization and classification in a scikit-learn `Pipeline` and fit the pipeline separately inside each split/fold.

This is an educational text-classification project. Its predictions and probability scores are not a substitute for fact-checking or reliable evidence about a claim's truth. Performance depends on the training data and may reflect its source, topic, or writing style.

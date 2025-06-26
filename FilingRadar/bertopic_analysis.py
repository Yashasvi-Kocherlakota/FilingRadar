import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Load the CSV
df = pd.read_csv("output/finbert_results.csv")
negatives = df[df["sentiment"] == "negative"]["sentence"].dropna().tolist()

# Step 2: Initialize BERTopic
topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)

# Optional: use a simpler vectorizer if memory is limited
# vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
# topic_model = BERTopic(vectorizer_model=vectorizer_model)

# Step 3: Fit the model
topics, probs = topic_model.fit_transform(negatives)

# Step 4: Get and display topics
topic_info = topic_model.get_topic_info()
print(topic_info.head())

# Step 5: Show example sentences per topic
for topic_id in topic_info["Topic"].head().tolist():
    if topic_id == -1: continue  # Skip outliers
    print(f"\n--- Topic {topic_id} ---")
    for sentence in topic_model.get_representative_docs(topic_id)[:5]:
        print(f"- {sentence[:200]}...")


# Save all topics to CSV
topics_df = topic_info.copy()
topics_df["Top Words"] = topics_df["Representation"].apply(lambda x: ", ".join(x[:10]))
topics_df.to_csv("output/bertopic_topics.csv", index=False)

topic_labels = {
    0: "Operational Risk & Regulatory Burden",
    1: "Market Volatility & Fee Compression",
    -1: "Uncategorized / Noise"
}
topics_df["Pain Point"] = topics_df["Topic"].map(topic_labels)
topics_df.to_csv("output/mapped_topics.csv", index=False)

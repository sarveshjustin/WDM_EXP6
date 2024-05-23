
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    sorted_indexes = similarity_scores.argsort()[0][::-1]
    results = [(documents[i], similarity_scores[0, i]) for i in sorted_indexes]
    return results



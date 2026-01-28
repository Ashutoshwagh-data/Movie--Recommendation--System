🎬 Movie Recommendation System
Live Project Link
👉 https://movie-recommendation-system-ac585jhtovt2s3kkbbfwyf.streamlit.app/

A content-based Movie Recommendation System built using Python and Streamlit.
The application recommends movies based on semantic similarity of movie titles and genres, using pre-trained embeddings and cosine similarity.

🚀 Features
🔍 Search movie by name using free-text input (partial match supported)
🎯 Recommends similar movies based on content similarity
🗂️ Filter recommendations by release year
🎭 Filter recommendations by genres
🔢 Select number of recommendations dynamically
⚡ Fast and lightweight with cached embeddings
🌐 Deployed using Streamlit Cloud
🧠 Uses embedding-based similarity (not keyword matching)
🛠️ Tech Stack
Python
Pandas
NumPy
Streamlit
Scikit-learn
Sentence-Transformers
📁 Project Structure
movie-recommendation-system/ │ ├── app.py ├── movies.csv ├── requirements.txt └── README.md

📊 Dataset
Dataset sourced from Kaggle

Dataset Link:
👉 https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system

Dataset Columns Used
title – Movie title
Year – Release year
genres – Pipe-separated movie genres
The dataset is cleaned and preprocessed inside the application.

🧠 Recommendation Approach
This project implements a Content-Based Recommendation System:

Movie titles and genres are combined into a single textual feature
Text is converted into dense vector embeddings using a pre-trained SentenceTransformer model
Cosine similarity is used to measure similarity between movies
Movies with the highest similarity scores are recommended
Year and Genre filters are applied after similarity computation
This approach avoids cold-start issues and works without user ratings.

⚙️ Performance Optimization
Embeddings and model loading are cached using Streamlit’s caching mechanisms
Similarity computation is done efficiently using vector operations
This ensures fast response times even with large datasets.

☁️ Deployment (Streamlit Cloud)
Push the project to GitHub
Visit 👉 https://streamlit.io/cloud
Click New App
Select the repository
Set the main file as app.py
Deploy 🎉
✅ Example Movies to Test
Toy Story
Jumanji
Titanic
Inception
The Dark Knight
⚠️ Limitations
This is not a personalized recommender
No user ratings or interaction data
Recommendations are based purely on movie content
📌 Future Improvements
🔎 Fuzzy search and auto-suggestions
⭐ Hybrid recommendation (content + ratings)
🚀 Faster similarity search using FAISS
🎨 Improved UI and visual enhancements
📱 Mobile-friendly layout
🙌 Acknowledgements
Kaggle – for providing the dataset
Sentence-Transformers – for pre-trained embedding models
Streamlit – for rapid application deployment
📬 Contact
If you found this project useful or have suggestions for improvement, feel free to connect.

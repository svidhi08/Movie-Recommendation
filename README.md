<div align="center">
  <h1>🎬 Movie Recommendation System – Hybrid AI Recommender</h1>
  <p><strong>A smart movie recommendation web application using Hybrid Filtering.</strong></p>
</div>

<hr/>

<h2>📌 Project Overview</h2>

<p>
This project is a <strong>Hybrid Recommendation System</strong> combining:
</p>

<ul>
  <li><strong>Content-Based Filtering (CBF)</strong> – TF-IDF + Cosine Similarity</li>
  <li><strong>Collaborative Filtering (SVD)</strong> – Matrix Factorization for personalized recommendations</li>
</ul>

<p>
It first finds similar movies, then ranks them using collaborative filtering based on the user’s history.
</p>

<hr/>

<h2>🎯 Recommendation Strategy</h2>

<ul>
  <li><strong>Step 1:</strong> Select Top 15–30 similar movies using TF-IDF + Cosine Similarity</li>
  <li><strong>Step 2:</strong> Rank candidates using SVD collaborative filtering for the given user</li>
  <li><strong>Step 3:</strong> Apply global popularity adjustment and user “soulmate” logic to boost recommendations</li>
  <li><strong>Final Output:</strong> Personalized Top Recommendations per user</li>
</ul>

<hr/>

<h2>🔎 Content-Based Filtering</h2>

<ul>
  <li><strong>Technique:</strong> TF-IDF Vectorization</li>
  <li><strong>Similarity Metric:</strong> Cosine Similarity</li>
  <li><strong>Features Used:</strong> Movie Title + Genres</li>
  <li><strong>Model Saved As:</strong> content_similarity.pkl</li>
</ul>

<hr/>

<h2>🤝 Collaborative Filtering</h2>

<ul>
  <li><strong>Type:</strong> Model-Based Collaborative Filtering</li>
  <li><strong>Algorithm Used:</strong> SVD (Matrix Factorization)</li>
  <li><strong>Library:</strong> Surprise</li>
  <li><strong>Model Saved As:</strong> svd_model.pkl</li>
</ul>

<hr/>

<h2>🚀 Main Features</h2>

<ul>
  <li>Personalized recommendations per user ID</li>
  <li>"Continue the Story" section – detects sequels and series</li>
  <li>"More Like This" – Content-Based Recommendations</li>
  <li>"Others Also Enjoyed" – Collaborative Filtering with user-soulmate + popularity adjustment</li>
  <li>Efficient prediction using pre-saved artifacts</li>
  <li>Memory-optimized for deployment on free cloud tiers</li>
  <li>Flask-based Web Interface</li>
</ul>

<hr/>

<h2>🖼️ Application Preview</h2>

<div align="center">

<h3>🏠 Home Page</h3>
<img src="images/home.png" width="800"/>

<br/><br/>

<h3>🎯 Recommendation Page</h3>
<img src="images/recommend.png" width="800"/>

</div>

<h2>🛠️ Technologies Used</h2>

<ul>
  <li><strong>Python</strong></li>
  <li><strong>Flask</strong></li>
  <li><strong>Scikit-learn</strong></li>
  <li><strong>Surprise (SVD)</strong></li>
  <li><strong>Pandas & NumPy</strong></li>
</ul>

<hr/>

<h2>📖 How to Run</h2>

<ol>
  <li>Clone Repository:
    <pre>git clone https://github.com/your-username/your-repo-name.git</pre>
  </li>
  <li>Install Requirements:
    <pre>pip install -r requirements.txt</pre>
  </li>
  <li>Check Artifacts Folder:
    <pre>
/artifacts
 ├── content_similarity.pkl
 └── svd_model.pkl
    </pre>
  </li>
  <li>Run App:
    <pre>python app.py</pre>
  </li>
</ol>

<hr/>

<h2>⚠️ Notes</h2>

<ul>
  <li>Hybrid recommendation combines content similarity and collaborative filtering to personalize results.</li>
  <li>"Soulmate" logic boosts recommendations based on users with similar tastes.</li>
  <li>Global popularity adjustment prevents blockbuster bias in scoring.</li>
</ul>

<div align="center">
  <p><i>Developed by <b>Vidhi</b> 🎬</i></p>
</div>

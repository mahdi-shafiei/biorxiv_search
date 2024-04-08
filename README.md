### bioRxiv Manuscript Semantic Search (bMSS)

---

### Links to the tool (mirrored):

* https://biorxiv-search.streamlit.app/

* https://huggingface.co/spaces/dzyla/bmss

  ---

### Purpose:
The Manuscript Semantic Search (bMSS) is a Streamlit-based web application designed for semantic search of scientific manuscripts. It leverages the Hugging Face API and Sentence Transformers to encode search queries and perform semantic searches over a corpus of manuscript embeddings. The application aims to provide an efficient and user-friendly interface for querying scientific articles, supporting researchers in finding relevant literature based on semantic content rather than keyword matching alone.

### Details (autogenerated):
1. **Framework and Libraries**: The application is built using Streamlit for the web interface, leveraging Pandas for data manipulation, NumPy for numerical operations, PyTorch for tensor operations and embeddings normalization, and Plotly for visualizing search results. It uses the Hugging Face API for embedding generation and Sentence Transformers for additional NLP tasks.

2. **Embedding Generation and Normalization**:
    - Queries are sent to the Hugging Face API, which returns embeddings for the text. These embeddings are then normalized to unit length to ensure uniformity in vector space.

3. **Embedding Quantization**:
    - For storage efficiency and faster retrieval, embeddings are quantized to a specified precision ("ubinary" by default). This process involves converting the embeddings to a binary format, reducing memory usage while retaining essential semantic information.

4. **Semantic Search**:
    - The application performs semantic searches using FAISS (Facebook AI Similarity Search), a library for efficient similarity searching. This allows for rapid retrieval of the most semantically relevant manuscripts based on the query's embeddings.

5. **Data and Embeddings Loading**:
    - Manuscript data and their corresponding embeddings are loaded from stored files. Duplicate entries are removed based on manuscript titles to ensure uniqueness in the search corpus.

6. **User Interface and Experience**:
    - Users can input search queries and specify the number of results to display. The application provides a responsive interface for users to interact with, displaying search results along with details such as manuscript title, authors, abstract, and a link to the full text.

7. **Visualization**:
    - Search results are visualized using Plotly, displaying publication dates, normalized scores, and category distributions to help users understand the relevance and distribution of the search outcomes.

### Setup
**Install**
1. `conda create -n bmss python=3.11 -y`
2. `conda activate bmss`
3. `pip install -r requirements.txt`
4. `streamlit run streamlit_app.py`

**Update**
1. `bash run_auto_update.sh`
2. Or python update_database.py

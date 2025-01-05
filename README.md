# Analytics Vidhya Course Search

A smart search system for finding relevant free courses on Analytics Vidhya's platform using advanced natural language processing and vector similarity search.

## Project Overview

This project implements a smart search feature that helps users find the most relevant free courses from Analytics Vidhya's platform. It uses modern NLP techniques and vector similarity search to provide accurate and meaningful search results.

**Created by:** Satyam2192
**Last Updated:** 2025-01-05

### Features

- Smart semantic search using embeddings
- Real-time course data collection
- Vector similarity search using FAISS
- User-friendly web interface using Streamlit
- Efficient data storage and retrieval
- Customizable search parameters

## Technical Stack

- **Language:** Python 3.7+
- **Main Libraries:**
  - LangChain 
  - FAISS-CPU 
  - Sentence-Transformers
  - Streamlit
  - Pandas
  - BeautifulSoup4
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Database:** FAISS
- **Web Framework:** Streamlit

## Installation

```bash
# 1. Clone the repository:
git clone <repository-url>
cd smart_search

#2. Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate  
# .\venv\Scripts\activate  # On Windows

# 3. Install dependencies:
pip install -r requirements.txt

## Usage

# 1. Collect course data:
python3 src/data_collection.py

# 2. Build the search index:
python3 src/build_index.py

# 3. Run the Streamlit app:
streamlit run src/app.py
```

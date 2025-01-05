import streamlit as st
from build_index import CourseSearchSystem
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_search_system():
    """Initialize the course search system."""
    search_system = CourseSearchSystem()
    
    # Check if index exists
    index_path = Path('../embeddings')
    if not index_path.exists() or not any(index_path.iterdir()):
        logger.info("Building new search index...")
        try:
            search_system.build_index()
        except Exception as e:
            st.error(f"Error building index: {str(e)}")
            return None
    else:
        try:
            search_system.load_index()
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            logger.error(f"Failed to load index: {str(e)}")
            logger.info("Attempting to rebuild index...")
            try:
                search_system.build_index()
            except Exception as rebuild_error:
                st.error(f"Error rebuilding index: {str(rebuild_error)}")
                return None
    
    return search_system

def main():
    st.set_page_config(
        page_title="Analytics Vidhya Course Search",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Analytics Vidhya Course Search")
    st.write("""
    Welcome to the Analytics Vidhya Course Search tool! This smart search system helps you
    find the most relevant free courses based on your interests and learning goals.
    """)

    # Initialize search system
    search_system = initialize_search_system()
    
    if search_system is None:
        st.error("Failed to initialize search system. Please contact support.")
        return

    # Search interface
    query = st.text_input(
        "🔍 Search for courses",
        placeholder="Enter your search query (e.g., 'machine learning for beginners')"
    )

    # Search parameters
    col1, col2 = st.columns(2)
    with col1:
        k = st.slider(
            "Number of results",
            min_value=1,
            max_value=20,
            value=5
        )
    with col2:
        similarity_cutoff = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    if query and search_system:
        try:
            results = search_system.search_courses(
                query,
                k=k,
                similarity_cutoff=similarity_cutoff
            )

            if results:
                st.subheader(f"Found {len(results)} relevant courses:")
                
                for result in results:
                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            if result['image_url']:
                                try:
                                    st.image(
                                        result['image_url'],
                                        width=150
                                    )
                                except Exception:
                                    st.write("Image unavailable")
                                
                        with col2:
                            st.markdown(f"### [{result['title']}]({result['url']})")
                            st.write(f"Number of Lessons: {result['lesson_count']}")
                            st.write(f"Similarity Score: {result['similarity_score']:.3f}")
                            
                    st.markdown("---")
            else:
                st.warning("No courses found matching your query. Please try different search terms.")
        except Exception as e:
            st.error(f"An error occurred while searching: {str(e)}")
            logger.error(f"Search error: {str(e)}")

if __name__ == "__main__":
    main()
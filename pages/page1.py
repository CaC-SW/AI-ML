import streamlit as st
from rag import load_all_csvs, vectorstore, retrieve_top_k, query

st.set_page_config(page_title="Offline RAG Bot", layout="wide")
st.title("GenAI QnA RAG Bot ")

if "collection" not in st.session_state:
    with st.spinner("Loading and indexing CSV data..."):
        chunks = load_all_csvs("Dataset")
        st.session_state.collection = vectorstore(chunks)
        st.success("Vector DB created successfully!")

user_input = st.text_input("Ask your question:")
if user_input and st.session_state.collection:
    with st.spinner("Searching and generating answer..."):
        relevant_chunks = retrieve_top_k(user_input, st.session_state.collection)
        context = "Search relevant details regarding the question from the data. If similar companies data is not found, then give relevant generic information and specify that you don't have information regarding that specific company in your database.".join(relevant_chunks)
        final_prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_input}"
        answer = query(final_prompt)
        st.subheader("Answer")
        st.write(answer)

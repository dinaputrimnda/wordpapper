import streamlit as st
import tempfile

from app_functions import (
    extract_text_from_pdf,
    preprocess_text,
    build_sentence_cooccurrence_graph,
    build_cooccurrence_matrix,
    compute_pagerank,
    visualize_word_graph
)

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Word Graph & PageRank Paper",
    layout="wide"
)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("Pengaturan Analisis")

uploaded_file = st.sidebar.file_uploader(
    "Upload Paper (PDF)",
    type=["pdf"]
)

top_pr = st.sidebar.slider(
    "Jumlah Top PageRank",
    min_value=10,
    max_value=50,
    value=20,
    step=5
)

process_btn = st.sidebar.button("Proses Paper")

# ===============================
# MAIN PAGE
# ===============================
st.title("Analisis Word Graph & PageRank dari Paper PDF")

st.markdown(
    """
    Aplikasi untuk menampilkan hubungan kata (**Word Co-occurrence Graph**)  
    dan nilai **PageRank** sebagai sentralitas kata dalam paper ilmiah.
    """
)

st.info("Upload PDF dan klik **Proses Paper** untuk memulai analisis.")

# ===============================
# PROCESSING
# ===============================
if uploaded_file and process_btn:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Extract & preprocess
    raw_text = extract_text_from_pdf(pdf_path)
    sentences = preprocess_text(raw_text)

    # Build graph
    graph = build_sentence_cooccurrence_graph(
        sentences,
        top_n=top_pr * 3   # supaya graph cukup padat
    )

    matrix = build_cooccurrence_matrix(graph)
    pagerank_df = compute_pagerank(graph)

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("📄 Preview Isi Paper")
    st.text_area(
        "Teks hasil ekstraksi (ringkas):",
        raw_text[:2500],
        height=200
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🌐 Word Graph")
        fig = visualize_word_graph(graph)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Top PageRank Centrality")
        st.dataframe(pagerank_df.head(top_pr))

    st.subheader("📊 Co-occurrence Matrix")
    st.dataframe(matrix)

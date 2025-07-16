import pandas as pd
import streamlit as st
import subprocess


st.set_page_config(page_title="Survey NLP Analyzer", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="stSidebarCollapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Survey Text Analyzer")
st.markdown("Upload Spreadsheet Data")

uploaded_file = st.file_uploader("Upload your survey CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
    except Exception as e:
        st.error(f"Error reading file: {e}")

    col_options = df.columns.tolist()


    # 2. User picks columns
    text_cols = st.multiselect("Select text columns to consider for NLP analysis", col_options)

    group_col = st.selectbox("Select column to group responses by", col_options)

    st.markdown("### Missing Value Settings")
    custom_na_input = st.text_input(
        "Enter custom NA values (comma-separated)", value="-999"
    )
    custom_na_values = [v.strip() for v in custom_na_input.split(",") if v.strip()]
    
    if custom_na_values:
        df.replace(custom_na_values, pd.NA, inplace=True)
        st.success(f"Custom NA values replaced: {custom_na_values}")

    if text_cols:

        drop_na_rows = st.checkbox("Drop rows where all selected text fields are NA", value=True)
        if drop_na_rows:
            pre_drop_len = len(df)
            df = df.dropna(subset=text_cols, how='all')
            st.info(f"Dropped {pre_drop_len - len(df)} rows where all selected text fields were NA.")

        # Show head of selected text columns
        st.markdown("### Selected Text Columns")
        st.dataframe(df[text_cols])
        if st.button("Next: Preprocessing Step"):
            st.session_state.step_1_df = df[text_cols+[group_col]]
            st.session_state.text_cols = text_cols
            st.session_state.group_col = group_col
            st.switch_page("pages/preprocessing_page.py") 
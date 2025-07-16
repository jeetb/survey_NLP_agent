import pandas as pd
import streamlit as st
from textblob import TextBlob

st.set_page_config(page_title="Preprocessing Options", layout="wide", initial_sidebar_state="collapsed")

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


def print_sample_corrections(cleaned_df_sample, session_df_sample):
    st.markdown("### 🧾 Before vs. After Correction (in sample of size 10)")
    for i in range(len(cleaned_df_sample)):
        st.markdown(f"**Row {i+1}**")
        for col in cleaned_df_sample.columns:
            orig = session_df_sample.iloc[i].at[col]
            corr = cleaned_df_sample.iloc[i].at[col]

            if orig != corr:
                st.markdown(f"- **{col}**")
                st.markdown(f"<span style='color:gray;'>Original:</span> {orig}", unsafe_allow_html=True)
                st.markdown(f"<span style='color:green;'>Corrected:</span> {corr}", unsafe_allow_html=True)
            # else:
            #     st.markdown(f"- **{col}** — no changes")
        st.markdown("---")

#gpt autocorrect function
def gpt_correct(text, api_key):
    if not text or text.strip() == "":
        return text

    prompt = (
    f"Fix any spelling or grammar mistakes in this text and return ONLY the fixed text.\n\n"
    f"{text}\n\n"
    f"Only return the corrected version, nothing else."
)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )

        corrected = response.choices[0].message.content.strip()
        # Remove outer quotes if they wrap the entire string
        if corrected.startswith('"') and corrected.endswith('"'):
            corrected = corrected[1:-1].strip()
        elif corrected.startswith("'") and corrected.endswith("'"):
            corrected = corrected[1:-1].strip()
        return corrected

    except Exception as e:
        return f"[ERROR: {e}]"


st.title("Survey Text Analyzer")
st.markdown("Preprocessing Options")

session_df = st.session_state.step_1_df
text_cols = st.session_state.text_cols
group_col = st.session_state.group_col

api_key = st.text_input("Enter your OpenAI API key if you would like to apply GPT based autocrrection or summary on the next page.", type="password")

autocorrect_method = st.selectbox(
    "Choose autocorrection method:",
    options=[
        "None",
        "Each field before join (TextBlob - basic)",
        "Each field before join (GPT - advanced)",
    ]
)

#Right now it is head 5 but maybe get 10 random rows.
session_df_sample = session_df[text_cols].fillna('').astype(str).sample(10).copy()
cleaned_df_sample = session_df_sample.copy(deep=True)

#Textblob    
if autocorrect_method == "Each field before join (TextBlob - basic)":

    st.info("Applying basic spelling correction using TextBlob to a sample of 10 random rows...")
    
    for col in text_cols:
        st.write(f"Correcting: {col}")
        cleaned_df_sample[col] = cleaned_df_sample[col].apply(lambda x: str(TextBlob(x).correct()))

    print_sample_corrections(cleaned_df_sample, session_df_sample)



#GPT
elif autocorrect_method == "Each field before join (GPT - advanced)":

    #Ask for API key
    st.markdown("### OpenAI API Key")

    if api_key:
        st.info("Applying GPT-based correction per field...")
        try:
            for col in text_cols:
                st.write(f"Correcting: {col}")
                cleaned_df_sample[col] = cleaned_df_sample[col].apply(lambda x: gpt_correct(x,  api_key))
            print_sample_corrections(cleaned_df_sample, session_df_sample)
        except:
            st.markdown("Error connecting to OpenAI! Please check your API key carefully.")
    else:
        st.markdown("No correction is being applied. OpenAI API key must be entered to apply GPT correction.")


else:
    st.info("Applying no corrections. All your text data will be kept unchanged")

cleaned_df = session_df[text_cols].fillna('').astype(str).copy()


if st.button("Next: Insights"):
    if autocorrect_method != "None":

        with st.spinner("Applying autocorrect to all selected data...", show_time=True):
            if autocorrect_method == "Each field before join (TextBlob - basic)":
                for col in text_cols:
                    cleaned_df[col] = cleaned_df[col].apply(lambda x: str(TextBlob(x).correct()))
            elif autocorrect_method == "Each field before join (GPT - advanced)":
                if api_key:
                    for col in text_cols:
                        cleaned_df[col] = cleaned_df[col].apply(lambda x: gpt_correct(x, api_key))
        st.success("Autocorrection successfully applied!")

    
    with st.spinner(f"Grouping text data by {group_col}...", show_time=True):
        cleaned_df['combined_text'] = cleaned_df[text_cols].agg(" ".join, axis=1)
        cleaned_df[group_col] = session_df[group_col]
        grouped = cleaned_df.groupby(group_col)['combined_text'].apply(
                    lambda texts: " ".join(str(t) for t in texts if pd.notna(t))
                ).reset_index()
        

    st.session_state.step_2_df = grouped
    st.session_state.group_col = group_col
    if 'api_key' in locals():
        st.session_state.api_key = api_key
    st.switch_page("pages/insights_page.py") 
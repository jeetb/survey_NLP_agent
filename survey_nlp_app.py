import streamlit as st



# Define the pages
main_page = st.Page("pages/upload_page.py", title="Upload Survey Spreadsheet")
preprocessing_page = st.Page("pages/preprocessing_page.py", title="Preprocessing Options")
insights_page = st.Page("pages/insights_page.py", title="Insights")

# Set up navigation
pg = st.navigation([main_page,preprocessing_page, insights_page])

# Run the selected page
pg.run()
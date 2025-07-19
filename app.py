
# app.py (This code would be in a separate .py file for Streamlit deployment)
import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
import re # Import regex for better keyword extraction

# --- Configuration & Data Loading ---
try:
    # Attempt to get API key from environment variable (for Streamlit Cloud/local deployment)
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if not gemini_api_key:
        # This block is for local testing/Kaggle Notebooks where os.environ might not be set
        # For Streamlit Cloud, the key *must* be set as an environment variable.
        # Note: 'kaggle_secrets' is not available in Streamlit Cloud.
        # This part is primarily for local testing within Kaggle Notebook.
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            gemini_api_key = user_secrets.get_secret("GEMINI_API_KEY")
        except ImportError:
            st.error("GEMINI_API_KEY environment variable not found. Please set it via Streamlit Secrets or environment variables.")
            st.stop() # Stop the app if API key is missing

    if not gemini_api_key:
        st.error("GEMINI_API_KEY is empty. Please ensure it's correctly set in Kaggle Secrets or environment variables.")
        st.stop()

    genai.configure(api_key=gemini_api_key)
    print("Gemini API configured successfully.")
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# Load the Q&A dataset
@st.cache_data 
def load_qa_data():
    try:
        # In Colab, after copying, it's in the current working directory
        data_path = 'medquad.csv' 
        
        if not os.path.exists(data_path):
             st.error(f"Dataset not found at {data_path}. Please ensure it's uploaded to Colab session storage or copied to working directory.")
             return pd.DataFrame(columns=['Question', 'Answer', 'Focus Area'])

        qa_df = pd.read_csv(data_path)
        
        # --- IMPORTANT: Rename columns to 'Question', 'Answer', and 'Focus Area' for consistency ---
        expected_cols_mapping = {
            'question': 'Question', 
            'answer': 'Answer', 
            'focus_area': 'Focus Area'
        }
        
        for old_col, new_col in expected_cols_mapping.items():
            if old_col in qa_df.columns:
                qa_df.rename(columns={old_col: new_col}, inplace=True)
            elif new_col not in qa_df.columns: # Check if the new name exists directly
                st.error(f"CSV must contain '{old_col}' or '{new_col}' column. Please check your file header.")
                st.stop()
            
        print(f"Dataset loaded successfully from {data_path}. Shape: {qa_df.shape}")
        print("First 5 rows of your Q&A data (after potential renaming):")
        print(qa_df.head())
        return qa_df
    except FileNotFoundError:
        st.error(f"Dataset not found at {data_path}. Please ensure it's in the correct location.")
        return pd.DataFrame(columns=['Question', 'Answer', 'Focus Area']) # Include Focus Area in empty df
    except Exception as e:
        st.error(f"Error loading Q&A dataset: {e}")
        return pd.DataFrame(columns=['Question', 'Answer', 'Focus Area'])

qa_df = load_qa_data()

# Initialize the Gemini 2.5 Flash model
@st.cache_resource # Cache the model to avoid re-initializing on every rerun
def load_gemini_model():
    return genai.GenerativeModel('gemini-2.5-flash')

model = load_gemini_model()

# --- Helper function to identify focus area (REFINED) ---
def identify_focus_area_from_question(user_question, qa_dataframe):
    """
    Identifies the most probable focus area from the user's question based on keyword matching.
    Prioritizes direct focus area name matches within the user's question.
    Returns the identified focus area string or None if no strong medical match.
    """
    user_question_lower = user_question.lower()
    user_keywords = set(re.findall(r'\b\w+\b', user_question_lower))
    
    identified_focus_area = None
    max_focus_score = 0 
    
    unique_focus_areas = qa_dataframe['Focus Area'].dropna().unique() # Get unique non-null focus areas

    for focus_area_name in unique_focus_areas:
        focus_area_lower = str(focus_area_name).lower()
        
        current_score = 0
        
        # Strongest signal: if the user's question directly contains the focus area name
        if focus_area_lower in user_question_lower:
            current_score += 1000 # A very high bonus for direct containment
            current_score += len(focus_area_lower.split()) # Bonus for longer exact matches

        # Fallback/additional signal: count keyword overlap
        current_score += sum(1 for keyword in user_keywords if keyword in focus_area_lower)
        
        if current_score > max_focus_score:
            max_focus_score = current_score
            identified_focus_area = focus_area_name
            
    
    if max_focus_score < 5: 
        return None 
            
    return identified_focus_area


# --- Primary Retrieval function from dataset ---
def retrieve_context_from_dataset(user_question, qa_dataframe, top_n=3):
    """
    Retrieves relevant Q&A pairs from the dataframe by first identifying a focus area,
    then performing keyword matching within that area.
    Returns a string of context, which might be empty if no relevant data is found.
    """
    user_question_lower = user_question.lower()
    user_keywords = set(re.findall(r'\b\w+\b', user_question_lower))

    if qa_dataframe.empty or 'Question' not in qa_dataframe.columns or 'Answer' not in qa_dataframe.columns or 'Focus Area' not in qa_dataframe.columns:
        return "Knowledge base is empty or required columns are missing. Cannot retrieve context."

    
    identified_focus_area = identify_focus_area_from_question(user_question, qa_dataframe)

    # Filter the DataFrame based on the identified focus area
    filtered_df = qa_dataframe
    if identified_focus_area: # Only filter if a relevant focus area was identified
        print(f"Attempting to filter by identified focus area: '{identified_focus_area}'.")
        # Case-insensitive filtering for the focus area
        temp_filtered_df = qa_dataframe[qa_dataframe['Focus Area'].str.lower() == identified_focus_area.lower()]
        
        if not temp_filtered_df.empty:
            filtered_df = temp_filtered_df
        else:
            print(f"Filtering by '{identified_focus_area}' resulted in an empty DataFrame. Proceeding with full search.")


    # Find relevant Q&A pairs within the (potentially filtered) DataFrame
    found_qa_pairs = []
    
    # Prioritize exact question match first from the filtered DataFrame
    exact_match_row = filtered_df[filtered_df['Question'].str.lower() == user_question_lower]
    if not exact_match_row.empty:
        found_qa_pairs.append(f"Q: {exact_match_row.iloc[0]['Question']}\nA: {exact_match_row.iloc[0]['Answer']}")
        
        if top_n == 1:
            return "\n\n".join(found_qa_pairs)

    
    potential_matches = []
    for index, row in filtered_df.iterrows():
        qa_pair_string = f"Q: {row['Question']}\nA: {row['Answer']}"
        # Skip if this is the exact match we already added
        if qa_pair_string in found_qa_pairs:
            continue
            
        question_text_lower = str(row['Question']).lower()
        answer_text_lower = str(row['Answer']).lower()

        # Check for keyword overlap in question or answer
        keyword_overlap_count = sum(1 for keyword in user_keywords if keyword in question_text_lower or keyword in answer_text_lower)
        
        if keyword_overlap_count > 0: # Only add if there's at least one keyword overlap
            potential_matches.append((keyword_overlap_count, qa_pair_string))

    
    potential_matches.sort(key=lambda x: x[0], reverse=True)
    

    for count, qa_pair_string in potential_matches:
        if len(found_qa_pairs) < top_n:
            found_qa_pairs.append(qa_pair_string)
        else:
            break

    # Return the context string. It will be empty if no relevant pairs were found.
    return "\n\n".join(found_qa_pairs)


# --- Chatbot response generation function with smart Gemini fallback ---
def generate_response_with_context(user_prompt, qa_dataframe):
    if model is None:
        return "Chatbot not initialized. Please check API configuration."
    if qa_dataframe.empty:
        return "Knowledge base is empty. Please load your Q&A dataset."

    # 1. Attempt to retrieve context from the primary dataset
    context_from_dataset = retrieve_context_from_dataset(user_prompt, qa_dataframe)
    
    # Identify focus area for potential use in fallback or general guidance
    identified_focus_area = identify_focus_area_from_question(user_prompt, qa_dataframe)
    
    # Construct the base prompt for Gemini
    # This prompt is designed to allow Gemini to use its general knowledge if dataset context is weak.
    
    # If context from dataset is found, prioritize it.
    if context_from_dataset:
        print("Dataset context found. Using RAG prompt.")
        full_prompt = f"""
        You are a helpful question-answering assistant specializing in medical topics.
        Use the following provided information to answer the user's question.
        If the provided information is incomplete or does not fully answer the question,
        you may supplement it with your general knowledge about the topic,
        especially if a focus area like '{identified_focus_area}' (if identified) is relevant.
        Do not make up answers if you truly don't know, even with general knowledge.

        --- Provided Information ---
        {context_from_dataset}
        ---

        User's Question: {user_prompt}
        """
    else:
        # If no context from dataset, explicitly tell Gemini to use general knowledge
        # and guide it with the identified focus area if possible.
        print("No dataset context found. Falling back to Gemini's general knowledge.")
        if identified_focus_area: # Only guide with focus area if one was strongly identified
            full_prompt = f"""
            You are a helpful question-answering assistant specializing in medical topics.
            I could not find specific information in my knowledge base.
            Please answer the following question based on your general knowledge about '{identified_focus_area}'.
            If you truly don't know, state that you cannot answer.

            User's Question: {user_prompt}
            """
        else:
            
            full_prompt = f"""
            You are a helpful question-answering assistant.
            I could not find specific information in my medical knowledge base.
            Please answer the following question based on your general knowledge.
            If you truly don't know, state that you cannot answer.

            User's Question: {user_prompt}
            """

    try:
        response = model.generate_content(full_prompt)
        final_response = response.text
    except Exception as e:
        print(f"Error generating content from Gemini: {e}")
        if hasattr(e, 'response') and e.response.prompt_feedback.block_reason:
            final_response = "I'm sorry, I cannot answer that question due to content safety policies."
        else:
            final_response = "I'm having trouble understanding or responding right now. Please try again."
    
    return final_response

# --- Streamlit UI ---
st.set_page_config(page_title="Custom Q&A Chatbot with Gemini 2.5 Flash", layout="centered")

hide_github_icon_css = """
<style>
/* This targets the specific Streamlit element that contains the GitHub icon/link */
/* The class names might change slightly with Streamlit updates, but this is a common one */
.viewerBadge_container__1QSob {
    display: none !important;
}

/* You might also want to hide the "Deploy" button if it appears locally or in certain dev modes */
.stDeployButton {
    display: none !important;
}

/* To hide the "Made with Streamlit" footer (if it appears in your version) */
footer {
    visibility: hidden;
    height: 0%;
}

/* To hide the main menu (hamburger icon) if desired */
#MainMenu {
    visibility: hidden;
}

</style>
"""
st.markdown(hide_github_icon_css, unsafe_allow_html=True)


st.title("📚 Custom Q&A Chatbot (Powered by Gemini 2.5 Flash)")
st.markdown("Ask questions based on the medical knowledge in my dataset! If I don't have specific data, I'll try to use my general knowledge for the topic.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass the loaded dataframe to the response generation function
            full_response = generate_response_with_context(prompt, qa_df)
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

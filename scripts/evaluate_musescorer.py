import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUTS_FOLDER = "outputs"
AUT_NORM_DB_PATH = "databases/aut_human_norms.csv"

# MuseScorer Hybrid Settings
USE_LLM_JUDGE = True
JUDGE_MODEL = "llama-3.1-8b-instant"  # Fast, cheap model for Yes/No judgments
TOP_K_CANDIDATES = 5                  # How many semantic matches to send to the LLM
MIN_SEMANTIC_SIM = 0.30               # Skip LLM call if semantic similarity is entirely unrelated

CONDITION_ORDER = ["Standard", "Conventional", "Effective", "Boring", "Creative"]
CONDITION_PALETTE = {
    "Standard": "#4C72B0", "Conventional": "#DD8452", 
    "Effective": "#55A868", "Boring": "#8172B3", "Creative": "#C44E52"
}

# Load env variables for the LLM Judge
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ATT05522.env"))
try:
    llm_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
except:
    llm_client = None
    print("Warning: LLM Client failed to initialize. Rarity scoring will default to 0.0")

# ==========================================
# HELPER FUNCTIONS: DATA PREPARATION
# ==========================================
def parse_aut_list(text):
    if not isinstance(text, str): return []
    items = re.findall(r'(?:^|\n)\s*\**\d+[\.\)]\**\s*(.+)', text)
    return [re.sub(r'^\*+|\*+$', '', item).strip() for item in items if item.strip()]

def clean_model_name(name):
    if not isinstance(name, str): return "Unknown"
    return name.split('/')[-1]

def load_aut_normative_database(filepath, model):
    if not os.path.exists(filepath):
        print(f"Warning: AUT database not found at {filepath}.")
        return None
    
    df = pd.read_csv(filepath)
    # Group identical human responses together to get their frequency count
    grouped_df = df.groupby(['Item', 'Response']).size().reset_index(name='Count')
    
    db_dict = {}
    print("Embedding aggregated human normative database...")
    for item in grouped_df['Item'].unique():
        item_data = grouped_df[grouped_df['Item'] == item]
        responses = item_data['Response'].tolist()
        counts = item_data['Count'].tolist()
        embeddings = model.encode(responses)
        db_dict[item] = {
            "responses": responses,
            "counts": counts,
            "embeddings": embeddings,
            "total_humans": sum(counts)
        }
    return db_dict

def check_matches_with_llm(item, generated_idea, candidate_ideas):
    """Uses an LLM to judge if the generated idea functionally matches any retrieved human ideas."""
    if not llm_client: return []
    
    candidates_text = "\n".join([f"[{i}] {idea}" for i, idea in enumerate(candidate_ideas)])
    prompt = (
        f"You are an expert evaluator scoring a Divergent Thinking test.\n"
        f"The user was asked to find an alternative use for a '{item}'.\n\n"
        f"Generated Idea: '{generated_idea}'\n\n"
        f"Human Database Candidates:\n{candidates_text}\n\n"
        f"Which of the human candidates share the EXACT SAME functional intent as the Generated Idea? "
        f"Reply ONLY with a comma-separated list of the numbers inside the brackets (e.g., 0, 2). "
        f"If none match, reply with 'NONE'."
    )
    
    try:
        response = llm_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        reply = response.choices[0].message.content.strip().upper()
        if "NONE" in reply: return []
        
        # Extract the matched indices (e.g., "0, 2" -> [0, 2])
        matches = [int(num) for num in re.findall(r'\d+', reply)]
        return matches
    except Exception as e:
        print(f"LLM Judge Error: {e}")
        return []

# ==========================================
# HELPER FUNCTIONS: EVALUATION & PLOTTING
# ==========================================
def evaluate_musescorer_aut(eval_df, emb_model, aut_db):
    """Calculates Originality and LLM-Judged Rarity (MuseScorer method)."""
    
    print("Generating embeddings...")
    item_embeddings = emb_model.encode(eval_df['Item'].tolist())
    idea_embeddings = emb_model.encode(eval_df['Parsed_Ideas'].tolist())
    
    # 1. Originality (Semantic Distance from Prompt Item)
    similarities = np.sum(item_embeddings * idea_embeddings, axis=1) / (
        np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(idea_embeddings, axis=1)
    )
    eval_df['Originality_Score'] = 1 - similarities

    # 2. Rarity (Hybrid RAG + LLM Judge against Normative DB)
    print("Calculating Rarity via LLM Verification...")
    rarity_scores = []
    
    for idx, row in eval_df.iterrows():
        item = row['Item']
        idea = row['Parsed_Ideas']
        idea_emb = idea_embeddings[idx]
        
        if aut_db and item in aut_db:
            db = aut_db[item]
            # Step 1: Retrieval (Find Top K semantic matches)
            sims = cosine_similarity([idea_emb], db['embeddings'])[0]
            top_k_indices = np.argsort(sims)[-TOP_K_CANDIDATES:][::-1]
            
            candidate_ideas = []
            candidate_db_indices = []
            
            for i in top_k_indices:
                if sims[i] > MIN_SEMANTIC_SIM:
                    candidate_ideas.append(db['responses'][i])
                    candidate_db_indices.append(i)
            
            # Step 2: LLM Validation
            if candidate_ideas:
                matched_local_indices = check_matches_with_llm(item, idea, candidate_ideas)
                
                # Sum the frequency of the human ideas the LLM confirmed as exact matches
                total_match_frequency = 0
                for local_idx in matched_local_indices:
                    if local_idx < len(candidate_db_indices):
                        db_idx = candidate_db_indices[local_idx]
                        total_match_frequency += db['counts'][db_idx]
                
                # Rarity = 1 - (Confirmed Match Frequency / Total Human Sample Size)
                rarity = 1 - (total_match_frequency / db['total_humans'])
            else:
                # No semantic candidates found, highly rare
                rarity = 1.0
        else:
            # Fallback if no database
            rarity = 0.0
            
        rarity_scores.append(rarity)
        
        if idx > 0 and idx % 10 == 0:
            print(f"Scored {idx}/{len(eval_df)} ideas...")
            time.sleep(1) # Prevent Groq rate limits

    eval_df['Rarity_Score'] = rarity_scores
    return eval_df

def plot_musescorer_results(eval_df):
    """Generates and saves the MuseScorer AUT evaluation charts."""
    print("Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.barplot(
        data=eval_df, x='Short_Model', y='Originality_Score', hue='Condition',
        hue_order=CONDITION_ORDER, palette=CONDITION_PALETTE, ax=axes[0], errorbar='ci', capsize=0.05
    )
    axes[0].set_title('AUT Originality Score by Model')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Originality (Semantic Distance)')
    axes[0].tick_params(axis='x', rotation=25)
    
    sns.barplot(
        data=eval_df, x='Short_Model', y='Rarity_Score', hue='Condition',
        hue_order=CONDITION_ORDER, palette=CONDITION_PALETTE, ax=axes[1], errorbar='ci', capsize=0.05
    )
    axes[1].set_title('AUT Rarity Score (MuseScorer)')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Rarity (Inverse Frequency)')
    axes[1].tick_params(axis='x', rotation=25)
    
    # External Legend mapping
    for ax in axes:
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig("musescorer_aut_results.png", dpi=300)
    plt.close()
    print("Saved musescorer_aut_results.png")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f"Searching for AUT CSV files in '{OUTPUTS_FOLDER}/'...")
    search_pattern = os.path.join(OUTPUTS_FOLDER, "**", "*alternative_uses_task*.csv")
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        print("Error: No AUT CSV files found.")
        return
        
    print(f"Found {len(all_files)} AUT files. Loading data...")
    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    df = df[~df['Response'].astype(str).str.startswith('Error:')].copy()
    
    df = df[df['Task'].astype(str).str.strip() == "Alternative Uses Task"].copy()
    
    if df.empty:
        print("No valid AUT responses found in the loaded CSVs.")
        return

    df['Short_Model'] = df['Model'].apply(clean_model_name)
    
    print("Loading MPNet embedding model...")
    emb_model = SentenceTransformer('all-mpnet-base-v2')
    aut_db = load_aut_normative_database(AUT_NORM_DB_PATH, emb_model) if USE_LLM_JUDGE else None

    # Parse the lists
    df['Parsed_Ideas'] = df['Response'].apply(parse_aut_list)
    eval_df = df.explode('Parsed_Ideas').dropna(subset=['Parsed_Ideas']).reset_index(drop=True)

    # Evaluate the metrics using the helper function
    eval_df = evaluate_musescorer_aut(eval_df, emb_model, aut_db)

    # Plot the results using the helper function
    plot_musescorer_results(eval_df)

    eval_df.to_csv("musescorer_aut_scored_results.csv", index=False)
    print("MuseScorer AUT Evaluation Complete. Saved charts and 'musescorer_aut_scored_results.csv'.")

if __name__ == "__main__":
    main()
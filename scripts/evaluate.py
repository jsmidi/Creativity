import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUTS_FOLDER = "outputs"
SIMILARITY_THRESHOLD = 0.85

# External Database Settings
USE_EXTERNAL_DATABASE = True
DAT_FREQ_DB_PATH = "databases/dat_word_frequencies.csv"  # Expected cols: ['Word', 'Frequency']
AUT_NORM_DB_PATH = "databases/aut_human_norms.csv"       # Expected cols: ['Item', 'Response']

# 5 Conditions configured with specific ordering and colors
CONDITION_ORDER = ["Standard", "Conventional", "Effective", "Boring", "Creative"]
CONDITION_PALETTE = {
    "Standard": "#4C72B0",       # Blue
    "Conventional": "#DD8452",   # Orange
    "Effective": "#55A868",      # Green
    "Boring": "#8172B3",         # Purple
    "Creative": "#C44E52"        # Red
}

# ==========================================
# PARSING & HELPER FUNCTIONS
# ==========================================
def parse_aut_list(text):
    """Extracts text immediately following a numbered list (e.g., '1. ', '2) ')."""
    if not isinstance(text, str): 
        return []
    # Captures the text on the line immediately following a digit and a dot/parenthesis
    items = re.findall(r'(?:^|\n)\s*\**\d+[\.\)]\**\s*(.+)', text)
    return [re.sub(r'^\*+|\*+$', '', item).strip() for item in items if item.strip()]

def parse_dat_words(text):
    """Extracts just the first alphabetical word after a numbered list."""
    if not isinstance(text, str): 
        return []
    # Captures only the alphabetical characters immediately following the number
    words = re.findall(r'(?:^|\n)\s*\**\d+[\.\)]\**\s*\*?([a-zA-Z\-]+)', text)
    return [w.lower().strip() for w in words if w.strip()]

def clean_model_name(name):
    """Shortens model IDs for cleaner plot axes (e.g., 'meta-llama/Llama-3.1-8B' -> 'Llama-3.1-8B')."""
    if not isinstance(name, str): 
        return "Unknown"
    return name.split('/')[-1]

# ==========================================
# DATABASE LOADERS
# ==========================================
def load_dat_frequency_database(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: DAT database not found at {filepath}. Using fallback internal scoring.")
        return None
    df = pd.read_csv(filepath)
    return dict(zip(df['Word'].str.lower(), df['Frequency']))

def load_aut_normative_database(filepath, model):
    if not os.path.exists(filepath):
        print(f"Warning: AUT database not found at {filepath}. Using fallback internal scoring.")
        return None
    df = pd.read_csv(filepath)
    db_dict = {}
    print("Embedding external AUT normative database...")
    for item in df['Item'].unique():
        item_responses = df[df['Item'] == item]['Response'].tolist()
        db_dict[item] = model.encode(item_responses)
    return db_dict

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f"Searching for CSV files recursively in '{OUTPUTS_FOLDER}/'...")
    
    all_files = glob.glob(os.path.join(OUTPUTS_FOLDER, "**", "*.csv"), recursive=True)
    
    if not all_files:
        print(f"Error: No CSV files found in '{OUTPUTS_FOLDER}'.")
        return
        
    print(f"Found {len(all_files)} CSV files. Combining them...")
    df_list = []
    for file in all_files:
        try:
            temp_df = pd.read_csv(file)
            # Filter out rows that are API error messages
            if 'Response' in temp_df.columns:
                temp_df = temp_df[~temp_df['Response'].astype(str).str.startswith('Error:')].copy()
            df_list.append(temp_df)
        except Exception as e:
            print(f"Warning: Could not read {file}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    df['Short_Model'] = df['Model'].apply(clean_model_name)
    
    print("Loading MPNet embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')

    # Load External DBs
    dat_db = None
    aut_db = None
    if USE_EXTERNAL_DATABASE:
        dat_db = load_dat_frequency_database(DAT_FREQ_DB_PATH)
        aut_db = load_aut_normative_database(AUT_NORM_DB_PATH, model)

    all_scored_data = []

    # Iterate through each unique task
    for task_type in df['Task'].unique():
        print(f"\n{'='*50}\nProcessing Task: {task_type}\n{'='*50}")
        task_df = df[df['Task'] == task_type].copy()
        safe_name = task_type.replace(" ", "_").lower()

        # ----------------------------------------------------
        # SCORING: DIVERGENT ASSOCIATION TASK
        # ----------------------------------------------------
        if task_type == "Divergent Association Task":
            task_df['Parsed_Words'] = task_df['Response'].apply(parse_dat_words)
            
            # For fallback internal rarity calculation
            all_dat_words = []
            word_to_row_map = []
            for idx, words in enumerate(task_df['Parsed_Words']):
                for w in words:
                    all_dat_words.append(w)
                    word_to_row_map.append(idx)
                    
            if all_dat_words and dat_db is None:
                all_word_embs = model.encode(all_dat_words)
                sim_mat_global = cosine_similarity(all_word_embs)
                total_dat_words = len(all_dat_words)
                global_rarities = [1 - (np.sum(sim_mat_global[i] > SIMILARITY_THRESHOLD) / total_dat_words) for i in range(total_dat_words)]

            dat_originality, dat_rarity = [], []
            for idx, words in enumerate(task_df['Parsed_Words']):
                if len(words) < 2:
                    dat_originality.append(0.0)
                    dat_rarity.append(0.0)
                    continue
                
                # Originality (Semantic distance between words in response)
                w_emb = emb_model.encode(words)
                sm = cosine_similarity(w_emb)
                up_tri = np.triu_indices_from(sm, k=1)
                
                official_dat_score = (1 - np.mean(sm[up_tri])) * 100
                dat_originality.append(official_dat_score)
                
                # Rarity
                if dat_db is not None:
                    # EXTERNAL DATABASE: Rarity based on normative frequency
                    word_rarities = [1 - dat_db.get(w, 0.0001) for w in words]
                    dat_rarity.append(np.mean(word_rarities))
                else:
                    # FALLBACK: Rarity based on internal dataset frequency
                    word_indices = [i for i, row_idx in enumerate(word_to_row_map) if row_idx == idx]
                    dat_rarity.append(np.mean([global_rarities[i] for i in word_indices]) if word_indices else 0.0)
                    
            task_df['Originality_Score'] = dat_originality
            task_df['Rarity_Score'] = dat_rarity
            eval_df = task_df

        # ----------------------------------------------------
        # SCORING: AUT, METAPHOR, SCI HYPO
        # ----------------------------------------------------
        else:
            if task_type == "Alternative Uses Task":
                task_df['Parsed_Ideas'] = task_df['Response'].apply(parse_aut_list)
                eval_df = task_df.explode('Parsed_Ideas').dropna(subset=['Parsed_Ideas']).reset_index(drop=True)
            else:
                eval_df = task_df.copy().reset_index(drop=True)
                eval_df['Parsed_Ideas'] = eval_df['Response'].apply(lambda x: str(x).strip())

            item_embeddings = model.encode(eval_df['Item'].tolist())
            idea_embeddings = model.encode(eval_df['Parsed_Ideas'].tolist())
            
            # Originality (Distance from the prompt item)
            similarities = np.sum(item_embeddings * idea_embeddings, axis=1) / (
                np.linalg.norm(item_embeddings, axis=1) * np.linalg.norm(idea_embeddings, axis=1)
            )
            eval_df['Originality_Score'] = 1 - similarities

            # Rarity (External Database vs Internal Database)
            eval_df['Rarity_Score'] = 0.0
            for item in eval_df['Item'].unique():
                item_indices = eval_df[eval_df['Item'] == item].index
                item_specific_embeddings = idea_embeddings[item_indices]
                
                if aut_db is not None and item in aut_db:
                    # EXTERNAL DATABASE: Compare against human normative responses
                    reference_embeddings = aut_db[item]
                    pairwise_sims = cosine_similarity(item_specific_embeddings, reference_embeddings)
                    total_reference = len(reference_embeddings)
                    
                    item_inv_freqs = [
                        1 - (np.sum(pairwise_sims[i] > SIMILARITY_THRESHOLD) / total_reference)
                        for i in range(len(item_indices))
                    ]
                else:
                    # FALLBACK: Compare against other AI responses for this run
                    pairwise_sims = cosine_similarity(item_specific_embeddings)
                    total_ideas = len(item_indices)
                    
                    item_inv_freqs = [
                        1 - (np.sum(pairwise_sims[i] > SIMILARITY_THRESHOLD) / total_ideas)
                        for i in range(total_ideas)
                    ]
                    
                eval_df.loc[item_indices, 'Rarity_Score'] = item_inv_freqs
                
        all_scored_data.append(eval_df)

        # ----------------------------------------------------
        # PLOTTING: MULTI-MODEL COMPARISON PER TASK
        # ----------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.barplot(
            data=eval_df, x='Short_Model', y='Originality_Score', hue='Condition',
            hue_order=CONDITION_ORDER, palette=CONDITION_PALETTE, ax=axes[0], errorbar='ci', capsize=0.05
        )
        axes[0].set_title(f'{task_type}\nOriginality Score by Model')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Originality (Semantic Distance)')
        axes[0].tick_params(axis='x', rotation=25)
        
        sns.barplot(
            data=eval_df, x='Short_Model', y='Rarity_Score', hue='Condition',
            hue_order=CONDITION_ORDER, palette=CONDITION_PALETTE, ax=axes[1], errorbar='ci', capsize=0.05
        )
        axes[1].set_title(f'{task_type}\nRarity Score by Model')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Rarity (Inverse Frequency)')
        axes[1].tick_params(axis='x', rotation=25)
        
        # External Legend mapping
        for ax in axes:
            ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        plt.savefig(f"task_{safe_name}_all_models.png", dpi=300)
        plt.close()
        print(f"Saved task_{safe_name}_all_models.png")

    # ==========================================
    # AGGREGATED PLOT (ALL MODELS x ALL TASKS)
    # ==========================================
    print("\nGenerating Cross-Model Macro Aggregate Plots...")
    master_df = pd.concat(all_scored_data, ignore_index=True)
    
    # Macro-average across tasks per model and condition to avoid AUT sample-count bias
    macro_means = master_df.groupby(['Short_Model', 'Task', 'Condition'])[['Originality_Score', 'Rarity_Score']].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.barplot(
        data=macro_means, x='Short_Model', y='Originality_Score', hue='Condition',
        hue_order=CONDITION_ORDER, palette=CONDITION_PALETTE, ax=axes[0], errorbar='ci', capsize=0.05
    )
    axes[0].set_title('Macro-Aggregate Originality\n(Across All 4 Tasks)')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Mean Originality')
    axes[0].tick_params(axis='x', rotation=25)
    
    sns.barplot(
        data=macro_means, x='Short_Model', y='Rarity_Score', hue='Condition',
        hue_order=CONDITION_ORDER, palette=CONDITION_PALETTE, ax=axes[1], errorbar='ci', capsize=0.05
    )
    axes[1].set_title('Macro-Aggregate Rarity\n(Across All 4 Tasks)')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Mean Rarity')
    axes[1].tick_params(axis='x', rotation=25)
    
    for ax in axes:
        ax.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("master_model_comparison_all_tasks.png", dpi=300)
    plt.close()
    
    master_df.to_csv("all_tasks_scored_combined.csv", index=False)
    macro_means.to_csv("macro_averages_by_model_task.csv", index=False)
    print("Saved 'master_model_comparison_all_tasks.png' and summary CSVs successfully.")

if __name__ == "__main__":
    main()
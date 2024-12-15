import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import re

# Function to compute embeddings and store in a local file
def compute_embeddings(dataset_path, db_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a lightweight model for embeddings
    documents = []
    metadata = []

    for story_file in os.listdir(dataset_path):
        if story_file.endswith('.txt'):
            with open(os.path.join(dataset_path, story_file), 'r') as file:
                story_text = file.read()
                documents.append(story_text)
                metadata.append({"story_file": story_file})

    embeddings = model.encode(documents, convert_to_tensor=False)

    # Save embeddings and metadata
    with open(db_path, 'wb') as db_file:
        pickle.dump({"embeddings": embeddings, "metadata": metadata, "documents": documents}, db_file)

    print("Embeddings computed and stored locally.")

# Function to extract relationships from text
def extract_relations(text):
    relations = []
    pattern = r"(\b[A-Z][a-z]+\b).*?(\b[A-Z][a-z]+\b)"  # Simple pattern for proper nouns
    matches = re.findall(pattern, text)
    for match in matches:
        relations.append({"name": match[1], "relation": "Related"})
    return relations

# Function to retrieve character information
def get_character_info(character_name, db_path):
    if not os.path.exists(db_path):
        print("Error: Database not found. Run 'compute-embeddings' first.")
        return

    with open(db_path, 'rb') as db_file:
        data = pickle.load(db_file)

    embeddings = data['embeddings']
    metadata = data['metadata']
    documents = data['documents']

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([character_name], convert_to_tensor=False)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    best_idx = np.argmax(similarities)

    structured_info = {
        "name": character_name,
        "storyTitle": metadata[best_idx]['story_file'].replace('.txt', ''),
        "summary": documents[best_idx][:200],  # Simplistic summary: first 200 characters
        "relations": extract_relations(documents[best_idx]),
        "characterType": "Unknown"  # Placeholder for future enhancement
    }

    print(json.dumps(structured_info, indent=4))

# CLI Entry Point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Character Info Extraction")
    subparsers = parser.add_subparsers(dest="command")

    compute_parser = subparsers.add_parser("compute-embeddings", help="Compute embeddings for story files.")
    compute_parser.add_argument("dataset_path", help="Path to dataset folder containing story files.")
    compute_parser.add_argument("db_path", help="Path to store database.")

    info_parser = subparsers.add_parser("get-character-info", help="Get structured character info.")
    info_parser.add_argument("character_name", help="Name of the character to search for.")
    info_parser.add_argument("db_path", help="Path to database.")

    args = parser.parse_args()

    if args.command == "compute-embeddings":
        compute_embeddings(args.dataset_path, args.db_path)
    elif args.command == "get-character-info":
        get_character_info(args.character_name, args.db_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

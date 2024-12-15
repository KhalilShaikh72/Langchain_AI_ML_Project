import pickle

def get_character_info(character_name, db_path="vector_db.pkl"):
    """
    Retrieve character information from the vector database.

    Args:
        character_name (str): The name of the character to search for.
        db_path (str): Path to the vector database file.

    Returns:
        dict: Structured information about the character or an error message.
    """
    try:
        # Load the vector database
        with open(db_path, 'rb') as f:
            vector_db = pickle.load(f)

        # Search for the character in the database
        if character_name in vector_db:
            return vector_db[character_name]
        else:
            return {"error": "Character not found"}
    except FileNotFoundError:
        return {"error": "Vector database not found"}

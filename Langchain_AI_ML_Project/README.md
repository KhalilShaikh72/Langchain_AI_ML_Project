
# Character Information Extraction using Sentence Transformers

This project extracts structured information about characters from narrative stories. It uses Sentence Transformers for embedding generation and cosine similarity for matching character queries to story data.

## Features

- Compute embeddings for a collection of story files.
- Query character information and retrieve structured details, including relations, summaries, and roles.

## Setup Instructions

### Prerequisites

- Python 3.7 or above
- Install the required Python packages:
  ```bash
  pip install sentence-transformers scikit-learn numpy
  ```

### Dataset Preparation

1. Create a folder named `dataset` in your project directory.
2. Add text files (`.txt`) to the `dataset` folder, each containing a story.

### Running the Code

1. **Compute Embeddings**
   - Run the following command to compute embeddings for the stories in your dataset and save them to a database file:
     ```bash
     python langchain_ai_assignment.py compute-embeddings ./dataset ./vector_db.pkl
     ```

2. **Query Character Information**
   - To fetch details about a character, use the following command:
     ```bash
     python langchain_ai_assignment.py get-character-info "<Character Name>" ./vector_db.pkl
     ```
   - Example:
     ```bash
     python langchain_ai_assignment.py get-character-info "Mary" ./vector_db.pkl
     ```

### Output Format

The output will be a JSON object with the following structure:
```json
{
  "name": "Mary",
  "storyTitle": "story2",
  "summary": "In the small village of Everdale, young Mary discovered a magical amulet. The amulet, a gift from her grandmother, gave her the power to communicate with animals.",
  "relations": [
    {"name": "Jack", "relation": "Related"}
  ],
  "characterType": "Unknown"
}
```

### Edge Cases

- If the character is not found, the program will print an error message.
- Ensure the dataset and database paths are correct when running commands.

## Project Structure

- `langchain_ai_assignment.py`: Main script containing all functionality.
- `dataset/`: Folder containing story files.
- `vector_db.pkl`: Generated database file storing embeddings and metadata.

## Contributing

Feel free to fork the repository and make enhancements. Open a pull request for suggestions or fixes.

## License

This project is licensed under the MIT License.

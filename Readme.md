# Document Segmentation and Embedding Storage

This project provides a pipeline to preprocess text and layout from an image, segment the document into meaningful chunks, generate embeddings for these chunks, and store them in ChromaDB. It also includes functionality to retrieve document chunks based on query embeddings.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Preprocessing Text and Layout](#preprocessing-text-and-layout)
  - [Segmenting the Document](#segmenting-the-document)
  - [Generating Embeddings](#generating-embeddings)
  - [Storing in ChromaDB](#storing-in-chromadb)
  - [Retrieving Chunks](#retrieving-chunks)
- [Example](#example)

## Installation

1. Update your package lists:
    ```sh
    !apt-get update
    ```

2. Install Tesseract OCR:
    ```sh
    !apt-get install tesseract-ocr
    ```

3. Install the required Python packages:
    ```sh
    !pip install pytesseract transformers chromadb sentence-transformers
    ```

## Usage

### Preprocessing Text and Layout

This function extracts words and their bounding boxes from an image using Tesseract OCR.

```python
from PIL import Image
import pytesseract

# Setting the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def preprocess_text_and_layout(image):
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    bboxes = []
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip() != "":
            words.append(ocr_data['text'][i])
            left, top, width, height = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            bboxes.append([left, top, left + width, top + height])
    return words, bboxes
```
### Segmenting the Document

This function segments the document into chunks using the LayoutLMv3 model.

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import torch.nn.functional as F

def segment_document(image, words, bboxes):
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

    encoding = processor(images=image, text=words, boxes=bboxes, return_tensors="pt", padding="max_length", truncation=True)
    outputs = model(**encoding)

    # Extract logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)

    # Get the predicted class for each token
    predictions = torch.argmax(probabilities, dim=-1)

    # Segment the text into chunks based on predicted labels
    chunks = []
    current_chunk = []
    for word, label in zip(words, predictions[0].tolist()):
        if label == 1:  # Assuming 1 represents the beginning of a new chunk (adjust based on actual labels)
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```
### Generating Embeddings

This function generates embeddings for document chunks using Sentence Transformers.

```python
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks):
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(chunks)
    return [embedding.tolist() for embedding in embeddings]
```
### Storing in ChromaDB
This function stores document chunks and their embeddings in ChromaDB.

```python
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings())

# Retrieve or create a collection in ChromaDB
collection_name = "document_chunks"
try:
    collection = chroma_client.create_collection(name=collection_name)
except Exception as e:
    if "already exists" in str(e):
        collection = chroma_client.get_collection(name=collection_name)
    else:
        raise e

def store_in_chromadb(chunks, embeddings, metadata):
    for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata)):
        collection.add(ids=[f"{meta['source_document_id']}_chunk_{i}"], documents=[chunk], embeddings=[embedding], metadatas=[meta])
```
### Retrieving Chunks
This function retrieves document chunks based on a query embedding.

```python
def retrieve_chunks(query, top_k=5):
    # Generate the embedding for the query
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode(query).tolist()

    # Retrieve similar chunks from ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    # Extract and display the retrieved chunks and their metadata
    retrieved_chunks = []
    for result in results["documents"]:
        chunk = result
        metadata = results["metadatas"][results["documents"].index(result)]
        retrieved_chunks.append((chunk, metadata))
    
    return retrieved_chunks
```


## Example

The following example demonstrates how to use the provided functions to preprocess an image, segment the document, generate embeddings, store them in ChromaDB, and retrieve similar chunks based on a query.

```python
from PIL import Image

# Load the image
image_path = 'path to your image'
image = Image.open(image_path)

# Preprocess the text and layout from the image
words, bboxes = preprocess_text_and_layout(image)

# Segment the document into chunks
chunks = segment_document(image, words, bboxes)

# Generate embeddings for the document chunks
embeddings = generate_embeddings(chunks)

# Prepare metadata for each chunk
metadata = [{"source_document_id": image_path, "chunk_number": i} for i in range(len(chunks))]

# Store the chunks, embeddings, and metadata in ChromaDB
store_in_chromadb(chunks, embeddings, metadata)

# Evaluate and confirm storage
print("Chunks and their embeddings stored in ChromaDB successfully.")

# Example query to find relevant chunks
query = "blood pressure measurements"

# Retrieve and display the top-k document chunks that match the query
retrieved_chunks = retrieve_chunks(query)
for i, (chunk, metadata) in enumerate(retrieved_chunks):
    print(f"Chunk {i+1}:")
    print(f"Text: {chunk}")
    print(f"Metadata: {metadata}")
    print("="*50)

print("Chunks retrieved from ChromaDB successfully.")

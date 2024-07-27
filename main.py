# Install necessary packages
!apt-get update
!apt-get install tesseract-ocr
!pip install pytesseract transformers chromadb sentence-transformers

import pytesseract
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import chromadb
from chromadb.config import Settings

# Setting the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Function to preprocess text and layout
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

# Function to segment document into chunks
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

# Function to generate embeddings for document chunks
def generate_embeddings(chunks):
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(chunks)
    return [embedding.tolist() for embedding in embeddings]

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

# Function to store chunks and embeddings in ChromaDB
def store_in_chromadb(chunks, embeddings, metadata):
    for i, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata)):
        collection.add(ids=[f"{meta['source_document_id']}_chunk_{i}"], documents=[chunk], embeddings=[embedding], metadatas=[meta])

# Function to retrieve document chunks from ChromaDB based on query embedding
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

# Example usage
image_path = '/content/PMC3576793_00004.jpg'  
image = Image.open(image_path)

# Preprocess the text and layout
words, bboxes = preprocess_text_and_layout(image)

# Segment the document
chunks = segment_document(image, words, bboxes)

# Generate embeddings for the chunks
embeddings = generate_embeddings(chunks)

# Prepare metadata for each chunk
metadata = [{"source_document_id": image_path, "chunk_number": i} for i in range(len(chunks))]

# Store chunks and embeddings in ChromaDB
store_in_chromadb(chunks, embeddings, metadata)

# Evaluation
print("Chunks and their embeddings stored in ChromaDB successfully.")

# Example query
query = "blood pressure measurements"  

# Retrieve and display the top-k document chunks
retrieved_chunks = retrieve_chunks(query)
for i, (chunk, metadata) in enumerate(retrieved_chunks):
    print(f"Chunk {i+1}:")
    print(f"Text: {chunk}")
    print(f"Metadata: {metadata}")
    print("="*50)

print("Chunks retrieved from ChromaDB successfully.")

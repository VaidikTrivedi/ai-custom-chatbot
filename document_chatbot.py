import os
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss.contrib.torch_utils
import faiss
import nltk
from nltk.tokenize import sent_tokenize
import logging
import sys
from datetime import datetime

# Configure logging with more verbose settings
log_filename = f'chatbot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Create console handler with a higher log level
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Create file handler which logs even debug messages
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set root logger level to DEBUG

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Test logging
logging.info("Starting Document Chatbot application...")

class DocumentChatbot:
    def __init__(self):
        logging.info("Initializing DocumentChatbot...")
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Check for GPU and handle AMD specifically
        self.device = "cpu"
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                if any(vendor in gpu_name for vendor in ['amd', 'radeon']):
                    # AMD GPU detected
                    self.device = "cuda"
                    torch.backends.cudnn.benchmark = True
                    logging.info(f"AMD GPU detected: {gpu_name}")
                else:
                    logging.info(f"GPU detected but not AMD: {gpu_name}")
            else:
                logging.info("No GPU detected, using CPU")
        except Exception as e:
            logging.warning(f"Error detecting GPU, falling back to CPU: {str(e)}")
            
        logging.info(f"Using device: {self.device}")
        
        # Initialize models with more powerful variants
        self.qa_model_name = "deepset/deberta-v3-large-squad2"
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        
        try:
            # Load models with device configuration
            self.qa_pipeline = pipeline(
                'question-answering', 
                model=self.qa_model_name,
                device=0 if self.device == "cuda" else -1,
                max_seq_length=1024,  # Increased from 512
                doc_stride=512        # Increased from 256
            )
            
            # Load and configure embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            if self.device == "cuda":
                self.embedding_model.to(self.device)
        except Exception as e:
            logging.warning(f"Error loading models on GPU, falling back to CPU: {str(e)}")
            self.device = "cpu"
            self.qa_pipeline = pipeline(
                'question-answering', 
                model=self.qa_model_name,
                device=-1,
                max_seq_length=1024,
                doc_stride=512
            )
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize document storage with improved context handling
        self.documents = []
        self.sentences = []
        self.sentence_embeddings = None
        self.faiss_index = None
        self.context_window = 25     # Increased from 15 to get more context
        self.batch_size = 16
        self.max_context_size = 20   # Increased from 10 to get more relevant contexts
        self.min_confidence_score = 0.05  # Reduced from 0.1 to include more answers
        
        logging.info("Models loaded successfully")

    def add_document(self, text):
        """Add a new document to the knowledge base"""
        self.documents.append(text)
        new_sentences = sent_tokenize(text)
        self.sentences.extend(new_sentences)
        self._update_embeddings()
        logging.info(f"Added document with {len(new_sentences)} sentences")

    def _update_embeddings(self):
        """Update the sentence embeddings and FAISS index with GPU acceleration"""
        try:
            # Process embeddings in batches
            embeddings_list = []
            for i in range(0, len(self.sentences), self.batch_size):
                batch = self.sentences[i:i + self.batch_size]
                with torch.no_grad():  # Reduce memory usage
                    batch_embeddings = self.embedding_model.encode(
                        batch,
                        normalize_embeddings=True,
                        device=self.device,
                        show_progress_bar=False
                    )
                    embeddings_list.append(batch_embeddings)
            
            self.sentence_embeddings = np.vstack(embeddings_list)
            
            # Initialize FAISS index
            vector_dimension = self.sentence_embeddings.shape[1]
            
            # Create FAISS index
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    res = faiss.StandardGpuResources()  # Initialize GPU resources
                    self.faiss_index = faiss.GpuIndexFlatIP(res, vector_dimension)
                except Exception as e:
                    logging.warning(f"Failed to create GPU FAISS index: {e}. Using CPU index.")
                    self.faiss_index = faiss.IndexFlatIP(vector_dimension)
            else:
                self.faiss_index = faiss.IndexFlatIP(vector_dimension)
                
            # Add vectors to the index
            self.faiss_index.add(np.array(self.sentence_embeddings).astype('float32'))
            
        except Exception as e:
            logging.error(f"Error in _update_embeddings: {str(e)}")
            raise

    def find_relevant_context(self, query, k=20):  # Increased from 10 to 20
        """Find the most relevant sentences using GPU acceleration"""
        with torch.no_grad():
            query_embedding = self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                device=self.device,
                show_progress_bar=False
            )
        
        # Get more similar contexts
        _, indices = self.faiss_index.search(
            np.array(query_embedding).astype('float32'), 
            min(k, len(self.sentences))  # Ensure k doesn't exceed available sentences
        )
        
        # Get extended context window around each relevant sentence
        relevant_contexts = []
        seen_indices = set()
        
        for idx in indices[0]:
            # Expand context window
            start_idx = max(0, idx - self.context_window)
            end_idx = min(len(self.sentences), idx + self.context_window + 1)
            
            context_chunk = []
            for context_idx in range(start_idx, end_idx):
                if context_idx not in seen_indices:
                    context_chunk.append(self.sentences[context_idx])
                    seen_indices.add(context_idx)
            
            if context_chunk:
                relevant_contexts.append(" ".join(context_chunk))
        
        # Join all contexts with clear separation
        return " ".join(relevant_contexts)

    def answer_question(self, question):
        """Answer a question using GPU acceleration with longer answers"""
        if not self.sentences:
            return "No documents have been added to the knowledge base yet."
        
        try:
            # Get expanded context
            context = self.find_relevant_context(question, k=self.max_context_size)
            
            # Split context into chunks if it's too long
            max_chunk_length = 1024  # Increased from 512
            context_chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sent_tokenize(context):
                sentence_length = len(sentence.split())
                if current_length + sentence_length > max_chunk_length:
                    context_chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            if current_chunk:
                context_chunks.append(" ".join(current_chunk))
            
            # Process each chunk and combine answers
            answers = []
            confidences = []
            
            for chunk in context_chunks:
                with torch.no_grad():
                    result = self.qa_pipeline(
                        question=question,
                        context=chunk,
                        max_answer_len=500,    # Increased from 200
                        max_seq_len=1024,      # Increased from 512
                        handle_impossible_answer=True,
                        batch_size=self.batch_size
                    )
                    if result['score'] > self.min_confidence_score:  # Using lower threshold
                        answers.append(result['answer'])
                        confidences.append(result['score'])
            
            if not answers:
                return {
                    'answer': "I'm not confident enough to provide an answer based on the available information.",
                    'confidence': 0.0,
                    'context': context
                }
            
            # Combine answers into a coherent response with better formatting
            combined_answer = self._format_long_answer(answers)
            avg_confidence = sum(confidences) / len(confidences)
            
            return {
                'answer': combined_answer,
                'confidence': round(avg_confidence, 4),
                'context': context
            }
            
        except Exception as e:
            logging.error(f"Error in question answering: {str(e)}")
            return {
                'answer': "An error occurred while processing your question.",
                'confidence': 0.0,
                'context': ""
            }

    def _format_long_answer(self, answers):
        """Format multiple answers into a coherent long response"""
        # Remove duplicates while preserving order
        unique_answers = []
        seen = set()
        for answer in answers:
            if answer.lower() not in seen:
                unique_answers.append(answer)
                seen.add(answer.lower())
        
        # If we have multiple answers, format them into paragraphs
        if len(unique_answers) > 1:
            # Join answers with proper spacing and connecting words
            formatted_answer = unique_answers[0]
            for i, answer in enumerate(unique_answers[1:], 1):
                # Add connecting phrases for smooth transition
                if i == len(unique_answers) - 1:
                    formatted_answer += ". Furthermore, " + answer
                else:
                    formatted_answer += ". Additionally, " + answer
        else:
            formatted_answer = unique_answers[0] if unique_answers else ""
        
        return formatted_answer

def main():
    # Example usage
    chatbot = DocumentChatbot()
    
    # Add sample document
    sample_text = """
    This is a sample document. You can add your own documents here.
    The chatbot will process them and answer questions based on their content.
    """
    chatbot.add_document(sample_text)
    
    # Example question
    question = "What will the chatbot do?"
    answer = chatbot.answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['confidence']}")

if __name__ == "__main__":
    main() 
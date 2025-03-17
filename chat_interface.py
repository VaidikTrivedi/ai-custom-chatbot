from document_chatbot import DocumentChatbot
import os
from PyPDF2 import PdfReader
import json
import requests
from bs4 import BeautifulSoup
import html2text
import logging

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def scrape_webpage(url):
    print(f"Scraping webpage: {url}")
    try:
        # Send request with a user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()

        # Convert HTML to clean text
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.ignore_tables = True
        clean_text = h.handle(str(soup))

        # Basic text cleaning
        clean_text = ' '.join(clean_text.split())
        return clean_text

    except requests.RequestException as e:
        logging.error(f"Error scraping {url}: {str(e)}")
        return None

def read_file(file_path):
    try:
        print(f"Reading file: {file_path}")
        document_text = ""
        if file_path.lower().endswith('.pdf'):
            # Handle PDF file
            reader = PdfReader(file_path)
            for page in reader.pages:
                document_text += page.extract_text() + "\n"
        else:
            # Handle text file
            with open(file_path, 'r', encoding='utf-8') as file:
                document_text = file.read()
                
        if not document_text.strip():
            print("\nWarning: The document appears to be empty.")
            return
            
        return document_text

    except FileNotFoundError:
        print("\nError: File not found. Please check the file path and try again.")
    except Exception as e:
        print(f"\nError reading file: {str(e)}")

def add_document(chatbot, data_sources):
    file_paths = data_sources['local_file_paths']
    file_content = ""
    for file_path in file_paths:
        file_content = file_content + read_file(file_path)
    for web_url in data_sources['web_urls']:
        content = scrape_webpage(web_url)
        if content:
            chatbot.add_document(content)
            print(f"Successfully added content from {web_url}")
        else:
            print(f"Failed to scrape content from {web_url}")
    return chatbot
            
def handle_question(chatbot):
    if not chatbot.sentences:
        print("\nNo documents added yet. Please add some documents first.")
        return
            
    question = input("\nEnter your question: ")
    result = chatbot.answer_question(question)
    
    print("\nAnswer:", result['answer'])
    print("Confidence:", result['confidence'])
    # print("\nRelevant context:", result['context'])

def main():
    print("Initializing Document Chatbot...")
    chatbot = DocumentChatbot()
    data_sources = json.load(open('data_sources.json'))    
    chatbot = add_document(chatbot, data_sources)
    handle_question(chatbot)

if __name__ == "__main__":
    main() 
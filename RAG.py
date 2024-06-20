from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd

class WebScraper:
    def __init__(self):
        self.visited_urls = set()
        self.scraped_data = {}
        self.nlp = spacy.load("en_core_web_sm")

    def scrape(self, url):
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        print(f"Scraping {url}")
        content = self.extract_content(soup)
        self.scraped_data[url] = content

    def extract_content(self, soup):
        content = []
        for paragraph in soup.find_all('p'):
            content.append(paragraph.get_text())
        return ' '.join(content)

    def preprocess_and_chunk(self):
        chunks = []
        chunk_id = 1
        for url, content in self.scraped_data.items():
            doc = self.nlp(content)
            sentences = [sent.text for sent in doc.sents]
            current_chunk = []
            current_length = 0

            for sent in sentences:
                sent_length = len(self.nlp(sent))
                if current_length + sent_length > 600:
                    chunks.append({'chunk_id': chunk_id, 'url': url, 'chunk': ' '.join(current_chunk)})
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0

                current_chunk.append(sent)
                current_length += sent_length

            if current_chunk:
                chunks.append({'chunk_id': chunk_id, 'url': url, 'chunk': ' '.join(current_chunk)})
                chunk_id += 1

        return pd.DataFrame(chunks)


if __name__ == "__main__":
    scraper = WebScraper()
    starting_url = "https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07"  # Replace with your starting URL
    scraper.scrape(starting_url)
    chunked_data_df = scraper.preprocess_and_chunk()
    print(chunked_data_df)

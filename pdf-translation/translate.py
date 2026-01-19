"""
XML Translation Script using OpenAI API
Translates text content in XML files while preserving structure and coordinates.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import sys
from openai import OpenAI
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install it with: pip install python-dotenv")
    pass


class XMLTranslator:
    def __init__(self, api_key: str, target_language: str = "Spanish", model: str = "gpt-4", max_workers: int = 5):
        """
        Initialize the XML Translator.
        
        Args:
            api_key: OpenAI API key
            target_language: Target language for translation (e.g., "Spanish", "French", "Hindi")
            model: OpenAI model to use (default: "gpt-4", can use "gpt-3.5-turbo" for faster/cheaper)
            max_workers: Number of concurrent threads for translation (default: 5)
        """
        self.client = OpenAI(api_key=api_key)
        self.target_language = target_language
        self.model = model
        self.max_workers = max_workers
        self.translation_cache = {}  # Cache to avoid re-translating same text
        self.cache_lock = Lock()  # Thread-safe cache access
        self.translation_count = 0
        self.count_lock = Lock()  # Thread-safe counter
        
    def translate_text(self, text: str, context: str = "") -> str:
        """
        Translate a single text string using OpenAI API.
        
        Args:
            text: Text to translate
            context: Optional context for better translation
            
        Returns:
            Translated text
        """
        # Skip empty or whitespace-only text
        if not text or not text.strip():
            return text
        
        # Check cache first (thread-safe)
        cache_key = f"{text}_{context}"
        with self.cache_lock:
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]
        
        try:
            # Create translation prompt
            if context:
                prompt = f"""Translate the following text to {self.target_language}. 
Context: {context}

Text to translate: {text}

Return ONLY the translated text, nothing else."""
            else:
                prompt = f"""Translate the following text to {self.target_language}.

Text to translate: {text}

Return ONLY the translated text, nothing else."""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate text accurately to {self.target_language}. Preserve formatting, numbers, and special characters. Return only the translation without any explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_tokens=2000
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Cache the translation (thread-safe)
            with self.cache_lock:
                self.translation_cache[cache_key] = translated_text
            
            # Increment counter (thread-safe)
            with self.count_lock:
                self.translation_count += 1
                if self.translation_count % 10 == 0:
                    print(f"  Progress: {self.translation_count} texts translated...")
            
            return translated_text
            
        except Exception as e:
            print(f"Error translating text: {str(e)}")
            return text  # Return original text if translation fails
    
    def translate_batch_concurrent(self, items: List[tuple], context: str = "") -> List[str]:
        """
        Translate multiple items concurrently using thread pool.
        
        Args:
            items: List of (index, text) tuples to translate
            context: Optional context for better translation
            
        Returns:
            List of (index, translated_text) tuples
        """
        results = []
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all translation tasks
            future_to_item = {
                executor.submit(self.translate_text, text, context): (idx, text) 
                for idx, text in items
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                idx, original_text = future_to_item[future]
                try:
                    translated_text = future.result()
                    results.append((idx, translated_text))
                except Exception as e:
                    print(f"Error translating item {idx}: {str(e)}")
                    results.append((idx, original_text))
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x[0])
        return [text for _, text in results]
    
        """
        Translate multiple texts in a single API call for efficiency.
        
        Args:
            texts: List of texts to translate
            context: Optional context for better translation
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # Filter out empty texts and keep track of indices
        non_empty_texts = []
        non_empty_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)
        
        if not non_empty_texts:
            return texts
        
        try:
            # Create batch translation prompt
            numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(non_empty_texts)])
            
            prompt = f"""Translate the following {len(non_empty_texts)} texts to {self.target_language}.
{f'Context: {context}' if context else ''}

Texts to translate:
{numbered_texts}

Return ONLY the translations in the same numbered format, nothing else."""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate texts accurately to {self.target_language}. Preserve formatting, numbers, and special characters. Return only the translations in numbered format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            translated_response = response.choices[0].message.content.strip()
            
            # Parse the numbered response
            translated_texts = []
            for line in translated_response.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i+1}.") for i in range(len(non_empty_texts))):
                    # Remove the number prefix
                    translated_text = line.split('.', 1)[1].strip() if '.' in line else line
                    translated_texts.append(translated_text)
            
            # If parsing failed, fall back to individual translations
            if len(translated_texts) != len(non_empty_texts):
                print("Warning: Batch translation parsing failed, falling back to individual translations")
                translated_texts = [self.translate_text(text, context) for text in non_empty_texts]
            
            # Reconstruct the full list with original empty texts
            result = list(texts)
            for i, idx in enumerate(non_empty_indices):
                if i < len(translated_texts):
                    result[idx] = translated_texts[i]
            
            return result
            
        except Exception as e:
            print(f"Error in batch translation: {str(e)}")
            # Fall back to individual translations
            return [self.translate_text(text, context) if text and text.strip() else text for text in texts]
    
    def translate_xml_element(self, element: ET.Element, page_context: str = ""):
        """
        Recursively translate text content in XML elements.
        
        Args:
            element: XML element to translate
            page_context: Context about the current page
        """
        # Translate text content in spans (most common text elements)
        if element.tag == "span" and element.text:
            print(f"Translating: {element.text[:50]}..." if len(element.text) > 50 else f"Translating: {element.text}")
            element.text = self.translate_text(element.text, page_context)
        
        # Translate table cell content
        elif element.tag == "cell" and element.text:
            print(f"Translating table cell: {element.text[:30]}...")
            element.text = self.translate_text(element.text, "table cell content")
        
        # For other elements with text, translate if needed
        elif element.text and element.text.strip() and element.tag not in ["document", "page", "text_blocks", "images", "tables", "drawings"]:
            element.text = self.translate_text(element.text, page_context)
        
        # Recursively process child elements
        for child in element:
            self.translate_xml_element(child, page_context)
    
    def translate_xml_file(self, input_xml_path: str, output_xml_path: str = None):
        """
        Translate an entire XML file using concurrent processing.
        
        Args:
            input_xml_path: Path to input XML file
            output_xml_path: Path to output XML file (optional)
        """
        if not os.path.exists(input_xml_path):
            print(f"Error: XML file '{input_xml_path}' not found!")
            return
        
        print(f"Loading XML file: {input_xml_path}")
        
        # Parse XML file
        tree = ET.parse(input_xml_path)
        root = tree.getroot()
        
        print(f"Translating to {self.target_language}...")
        print(f"Using model: {self.model}")
        print(f"Concurrent workers: {self.max_workers}")
        
        start_time = time.time()
        
        # Process each page
        for page_num, page in enumerate(root.findall("page"), 1):
            page_context = f"PDF page {page_num}"
            print(f"\nProcessing page {page_num}...")
            
            # Collect all texts to translate from this page
            all_items = []
            
            # Collect text spans
            text_blocks = page.find("text_blocks")
            if text_blocks is not None:
                spans = text_blocks.findall(".//span")
                if spans:
                    print(f"  Found {len(spans)} text spans")
                    for span in spans:
                        if span.text and span.text.strip():
                            all_items.append(('span', span, span.text, page_context))
            
            # Collect table cells
            tables = page.find("tables")
            if tables is not None:
                cells = tables.findall(".//cell")
                if cells:
                    print(f"  Found {len(cells)} table cells")
                    for cell in cells:
                        if cell.text and cell.text.strip():
                            all_items.append(('cell', cell, cell.text, "table content"))
            
            if not all_items:
                print(f"  No text to translate on page {page_num}")
                continue
            
            print(f"  Translating {len(all_items)} items concurrently...")
            
            # Translate all items concurrently
            items_to_translate = [(i, item[2]) for i, item in enumerate(all_items)]
            
            # Use concurrent translation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {}
                for idx, text in items_to_translate:
                    context = all_items[idx][3]
                    future = executor.submit(self.translate_text, text, context)
                    future_to_idx[future] = idx
                
                # Collect results
                translated_results = [None] * len(all_items)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        translated_results[idx] = future.result()
                    except Exception as e:
                        print(f"  Error translating item {idx}: {str(e)}")
                        translated_results[idx] = all_items[idx][2]  # Use original text
            
            # Update XML elements with translations
            for idx, (item_type, element, original_text, context) in enumerate(all_items):
                if translated_results[idx]:
                    element.text = translated_results[idx]
        
        elapsed_time = time.time() - start_time
        
        # Generate output filename if not provided
        if output_xml_path is None:
            base_name = os.path.splitext(input_xml_path)[0]
            output_xml_path = f"{base_name}_translated_{self.target_language.lower()}.xml"
        
        # Write translated XML to file
        xml_string = self.prettify_xml(root)
        with open(output_xml_path, "w", encoding="utf-8") as f:
            f.write(xml_string)
        
        print(f"\n✅ Translation complete!")
        print(f"Translated XML saved to: {output_xml_path}")
        print(f"Total translations: {self.translation_count}")
        print(f"Cached translations: {len(self.translation_cache)}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Average: {elapsed_time/self.translation_count:.2f} seconds per translation" if self.translation_count > 0 else "")
    
    @staticmethod
    def prettify_xml(elem):
        """Return a pretty-printed XML string."""
        rough_string = ET.tostring(elem, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


def main():
    """Main function to run the translator."""
    if len(sys.argv) < 3:
        print("Usage: python translate.py <input_xml> <target_language> [output_xml] [model] [max_workers]")
        print("\nExamples:")
        print("  python translate.py document.xml Spanish")
        print("  python translate.py document.xml French output_french.xml")
        print("  python translate.py document.xml Hindi output.xml gpt-3.5-turbo")
        print("  python translate.py document.xml Spanish output.xml gpt-4 10")
        print("\nSupported languages: Spanish, French, German, Hindi, Chinese, Japanese, etc.")
        print("Models: gpt-4 (default, better quality) or gpt-3.5-turbo (faster, cheaper)")
        print("Max workers: Number of concurrent threads (default: 5, recommended: 5-10)")
        sys.exit(1)
    
    input_xml = sys.argv[1]
    target_language = sys.argv[2]
    
    # Parse optional arguments
    output_xml = None
    model = "gpt-4"
    max_workers = 5
    
    # Check remaining arguments
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith("gpt"):
            model = arg
        elif arg.isdigit():
            max_workers = int(arg)
        elif arg.endswith(".xml"):
            output_xml = arg
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found!")
        print("\nPlease set it using one of these methods:")
        print("  1. Create a .env file in the same directory with:")
        print("     OPENAI_API_KEY=your-api-key")
        print("  2. Set environment variable:")
        print("     Windows: set OPENAI_API_KEY=your-api-key")
        print("     Linux/Mac: export OPENAI_API_KEY=your-api-key")
        print("\nNote: If using .env file, install python-dotenv:")
        print("  pip install python-dotenv")
        sys.exit(1)
    
    # Create translator and translate
    translator = XMLTranslator(
        api_key=api_key, 
        target_language=target_language, 
        model=model,
        max_workers=max_workers
    )
    translator.translate_xml_file(input_xml, output_xml)


if __name__ == "__main__":
    main()
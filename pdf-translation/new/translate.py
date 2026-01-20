"""
Fixed Enhanced XML Translation Script
Prevents prompt leakage and ensures clean translations while preserving all metadata.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import sys
from openai import OpenAI
from typing import List, Dict, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import copy
import re

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install it with: pip install python-dotenv")
    pass


class EnhancedXMLTranslator:
    def __init__(self, api_key: str, target_language: str = "Spanish", model: str = "gpt-4", max_workers: int = 5):
        """
        Initialize the Enhanced XML Translator.
        
        Args:
            api_key: OpenAI API key
            target_language: Target language for translation
            model: OpenAI model to use
            max_workers: Number of concurrent threads for translation
        """
        self.client = OpenAI(api_key=api_key)
        self.target_language = target_language
        self.model = model
        self.max_workers = max_workers
        self.translation_cache = {}
        self.cache_lock = Lock()
        self.translation_count = 0
        self.count_lock = Lock()
        
    def clean_translation(self, text: str, original: str) -> str:
        """
        Clean the translation to remove any prompt artifacts or instructions.
        
        Args:
            text: Translated text that may contain artifacts
            original: Original text for length comparison
            
        Returns:
            Cleaned translation
        """
        if not text:
            return text
        
        # Remove common prompt artifacts (multi-language)
        artifacts = [
            "Return ONLY the translated text, nothing else.",
            "Devuelve SOLO el texto traducido, nada más.",
            "Renvoie SEULEMENT le texte traduit, rien d'autre.",
            "Gib NUR den übersetzten Text zurück, sonst nichts.",
            "Restituisci SOLO il testo tradotto, nient'altro.",
            "Text to translate:",
            "Texto a traducir:",
            "Context:",
            "Contexto:",
        ]
        
        cleaned = text.strip()
        
        # Remove artifacts
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, "")
        
        # Remove leading/trailing special characters that might be artifacts
        cleaned = re.sub(r'^[■\s]+', '', cleaned)
        cleaned = re.sub(r'[■\s]+$', '', cleaned)
        
        # If the translation is suspiciously longer than 2x original, it might have artifacts
        if len(cleaned) > len(original) * 2.5:
            # Try to extract just the translation part
            lines = cleaned.split('\n')
            # Usually the translation is the longest line or last substantial line
            substantial_lines = [l.strip() for l in lines if len(l.strip()) > 10]
            if substantial_lines:
                cleaned = substantial_lines[-1]
        
        return cleaned.strip()
        
    def translate_text(self, text: str, context: str = "") -> str:
        """
        Translate a single text string using OpenAI API with improved prompt.
        
        Args:
            text: Text to translate
            context: Optional context for better translation (IGNORED to prevent leakage)
            
        Returns:
            Translated text (cleaned)
        """
        # Skip empty or whitespace-only text
        if not text or not text.strip():
            return text
        
        # Check cache first (thread-safe) - cache without context to avoid contamination
        cache_key = f"{text}"
        with self.cache_lock:
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]
        
        try:
            # Ultra-clean prompt - NO context strings to prevent any leakage
            system_prompt = f"You are a professional translator. Translate to {self.target_language}. Return ONLY the translation."
            user_prompt = text  # Just the text, nothing else!
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Clean the translation
            translated_text = self.clean_translation(translated_text, text)
            
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
    
    def preserve_element_attributes(self, source_elem: ET.Element, target_elem: ET.Element):
        """Copy all attributes from source element to target element."""
        for key, value in source_elem.attrib.items():
            target_elem.set(key, value)
    
    def translate_page_concurrent(self, page_elem: ET.Element, page_context: str) -> ET.Element:
        """
        Translate a page element with concurrent processing.
        """
        # Create new page element with preserved attributes
        new_page = ET.Element(page_elem.tag)
        self.preserve_element_attributes(page_elem, new_page)
        
        if page_elem.text:
            new_page.text = page_elem.text
        if page_elem.tail:
            new_page.tail = page_elem.tail
        
        # Collect all text items to translate
        text_items = []
        text_elements = []
        
        # Find all spans and cells in the page - NO CONTEXT to prevent leakage
        for text_block in page_elem.findall(".//text_blocks"):
            for span in text_block.findall(".//span"):
                if span.text and span.text.strip():
                    text_items.append((len(text_elements), span.text, "", span))  # Empty context!
                    text_elements.append(span)
        
        for tables in page_elem.findall(".//tables"):
            for cell in tables.findall(".//cell"):
                if cell.text and cell.text.strip():
                    text_items.append((len(text_elements), cell.text, "", cell))  # Empty context!
                    text_elements.append(cell)
        
        if text_items:
            print(f"  Found {len(text_items)} text items to translate")
            
            # Translate concurrently
            translations = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {}
                for idx, text, context, elem in text_items:
                    future = executor.submit(self.translate_text, text, context)
                    future_to_idx[future] = (idx, elem)
                
                # Collect results
                for future in as_completed(future_to_idx):
                    idx, elem = future_to_idx[future]
                    try:
                        translated = future.result()
                        translations[id(elem)] = translated
                    except Exception as e:
                        print(f"  Error translating item {idx}: {str(e)}")
                        translations[id(elem)] = elem.text
        
        # Now recursively build the new page structure with translations
        for child in page_elem:
            new_child = self.translate_element_with_cache(child, page_context, translations)
            new_page.append(new_child)
        
        return new_page
    
    def translate_element_with_cache(self, element: ET.Element, page_context: str, translations: Dict) -> ET.Element:
        """
        Translate element using pre-computed translations from cache.
        """
        # Create new element
        new_element = ET.Element(element.tag)
        
        # Preserve ALL attributes
        self.preserve_element_attributes(element, new_element)
        
        # Check if this element has a pre-computed translation
        elem_id = id(element)
        if elem_id in translations:
            new_element.text = translations[elem_id]
        elif element.text:
            # For non-translatable elements, preserve original text
            new_element.text = element.text
        
        # Preserve tail
        if element.tail:
            new_element.tail = element.tail
        
        # Recursively process children
        for child in element:
            new_child = self.translate_element_with_cache(child, page_context, translations)
            new_element.append(new_child)
        
        return new_element
    
    def translate_xml_file(self, input_xml_path: str, output_xml_path: str = None):
        """
        Translate an entire XML file while preserving ALL metadata.
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
        print("=" * 60)
        
        start_time = time.time()
        
        # Create new root with preserved attributes
        new_root = ET.Element(root.tag)
        self.preserve_element_attributes(root, new_root)
        
        # Preserve root text if any
        if root.text:
            new_root.text = root.text
        if root.tail:
            new_root.tail = root.tail
        
        # Process document-level metadata (preserve as-is)
        for child in root:
            if child.tag == "metadata":
                # Preserve metadata completely
                metadata_copy = copy.deepcopy(child)
                new_root.append(metadata_copy)
            elif child.tag == "page":
                # Process each page
                page_num = child.get("number", "?")
                print(f"\nProcessing page {page_num}...")
                
                # Translate page with concurrent processing
                translated_page = self.translate_page_concurrent(child, f"PDF page {page_num}")
                new_root.append(translated_page)
            else:
                # For any other top-level elements, preserve as-is
                child_copy = copy.deepcopy(child)
                new_root.append(child_copy)
        
        elapsed_time = time.time() - start_time
        
        # Generate output filename if not provided
        if output_xml_path is None:
            base_name = os.path.splitext(input_xml_path)[0]
            output_xml_path = f"{base_name}_translated_{self.target_language.lower()}.xml"
        
        # Write translated XML to file
        xml_string = self.prettify_xml(new_root)
        with open(output_xml_path, "w", encoding="utf-8") as f:
            f.write(xml_string)
        
        print("\n" + "=" * 60)
        print(f"✓ Translation complete!")
        print(f"✓ Translated XML saved to: {output_xml_path}")
        print(f"✓ Total translations: {self.translation_count}")
        print(f"✓ Cached translations: {len(self.translation_cache)}")
        print(f"✓ Time elapsed: {elapsed_time:.2f} seconds")
        print(f"✓ Average: {elapsed_time/self.translation_count:.2f} seconds per translation" if self.translation_count > 0 else "")
        print(f"✓ ALL METADATA PRESERVED + CLEAN TRANSLATIONS")
    
    @staticmethod
    def prettify_xml(elem):
        """Return a pretty-printed XML string."""
        rough_string = ET.tostring(elem, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


def main():
    """Main function to run the enhanced translator."""
    if len(sys.argv) < 3:
        print("Usage: python translate_enhanced_fixed.py <input_xml> <target_language> [output_xml] [model] [max_workers]")
        print("\nExamples:")
        print("  python translate_enhanced_fixed.py document_enhanced.xml Spanish")
        print("  python translate_enhanced_fixed.py document_enhanced.xml French output_french.xml")
        print("  python translate_enhanced_fixed.py document_enhanced.xml Hindi output.xml gpt-3.5-turbo")
        print("  python translate_enhanced_fixed.py document_enhanced.xml Spanish output.xml gpt-4 10")
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
        print("  1. Create a .env file with: OPENAI_API_KEY=your-api-key")
        print("  2. Set environment variable")
        sys.exit(1)
    
    # Create translator and translate
    translator = EnhancedXMLTranslator(
        api_key=api_key, 
        target_language=target_language, 
        model=model,
        max_workers=max_workers
    )
    translator.translate_xml_file(input_xml, output_xml)


if __name__ == "__main__":
    main()
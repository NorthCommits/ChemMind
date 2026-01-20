"""
ULTIMATE Surgical PDF Text Replacement Script
Handles text that is longer than original by using smart wrapping and sizing.

This version solves the Spanish > English length problem!
"""

import pymupdf
import xml.etree.ElementTree as ET
import os
import sys
from typing import Dict, List, Tuple


class UltimateSurgicalReplacer:
    def __init__(self, original_pdf_path: str, translated_xml_path: str):
        self.original_pdf_path = original_pdf_path
        self.translated_xml_path = translated_xml_path
        
        if not os.path.exists(original_pdf_path):
            raise FileNotFoundError(f"Original PDF not found: {original_pdf_path}")
        if not os.path.exists(translated_xml_path):
            raise FileNotFoundError(f"Translated XML not found: {translated_xml_path}")
        
        self.doc = pymupdf.open(original_pdf_path)
        self.tree = ET.parse(translated_xml_path)
        self.root = self.tree.getroot()
        
        self.stats = {
            'total_spans': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        print(f"✓ Loaded: {original_pdf_path}")
        print(f"✓ Loaded: {translated_xml_path}")
        print(f"✓ Pages: {len(self.doc)}\n")
    
    def parse_color_to_rgb(self, color_value) -> Tuple[float, float, float]:
        if not color_value or color_value == 'None':
            return (0, 0, 0)
        try:
            color_int = int(color_value) if isinstance(color_value, str) else color_value
            r = ((color_int >> 16) & 0xFF) / 255.0
            g = ((color_int >> 8) & 0xFF) / 255.0
            b = (color_int & 0xFF) / 255.0
            return (r, g, b)
        except:
            return (0, 0, 0)
    
    def extract_translation_data(self, page_num: int) -> List[Dict]:
        translation_data = []
        
        page_elem = None
        for page in self.root.findall('page'):
            if int(page.get('number', 0)) == page_num + 1:
                page_elem = page
                break
        
        if not page_elem:
            return translation_data
        
        text_blocks = page_elem.find('text_blocks')
        if not text_blocks:
            return translation_data
        
        for block in text_blocks.findall('block'):
            for line in block.findall('line'):
                for span in line.findall('span'):
                    text = span.text
                    if not text or not text.strip():
                        continue
                    
                    try:
                        bbox = (
                            float(span.get('x0', 0)),
                            float(span.get('y0', 0)),
                            float(span.get('x1', 0)),
                            float(span.get('y1', 0))
                        )
                        
                        translation_data.append({
                            'bbox': bbox,
                            'text': text,
                            'font': span.get('font', 'Helvetica'),
                            'size': float(span.get('size', 12)),
                            'color': self.parse_color_to_rgb(span.get('color')),
                            'bold': span.get('bold', 'False') == 'True',
                            'italic': span.get('italic', 'False') == 'True'
                        })
                    except Exception as e:
                        print(f"    ⚠ Parse error: {str(e)}")
                        continue
        
        return translation_data
    
    def replace_page_text(self, page_num: int):
        page = self.doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        
        translation_data = self.extract_translation_data(page_num)
        
        if not translation_data:
            print(f"  Page {page_num + 1}: No text found")
            return
        
        self.stats['total_spans'] += len(translation_data)
        print(f"  Page {page_num + 1}: {len(translation_data)} spans")
        print(f"    Size: {page_width:.0f} x {page_height:.0f}")
        
        # STEP 1: Redact all original text
        for item in translation_data:
            bbox = item['bbox']
            rect = pymupdf.Rect(bbox)
            try:
                page.add_redact_annot(rect, fill=(1, 1, 1), text="")
            except Exception as e:
                print(f"    ⚠ Redaction error: {str(e)}")
        
        try:
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE
            )
            print(f"    ✓ Redacted {len(translation_data)} spans")
        except Exception as e:
            print(f"    ✗ Redaction failed: {str(e)}")
            return
        
        # STEP 2: Insert translated text with insert_htmlbox (supports wrapping!)
        success_count = 0
        
        for idx, item in enumerate(translation_data):
            bbox = item['bbox']
            text = item['text']
            size = item['size']
            color = item['color']
            bold = item['bold']
            italic = item['italic']
            
            x0, y0, x1, y1 = bbox
            rect = pymupdf.Rect(x0, y0, x1, y1)
            
            # Build HTML with styling
            r, g, b = [int(c * 255) for c in color]
            font_weight = 'bold' if bold else 'normal'
            font_style = 'italic' if italic else 'normal'
            
            html = f'<span style="color: rgb({r},{g},{b}); font-weight: {font_weight}; font-style: {font_style};">{text}</span>'
            
            # CSS with automatic sizing
            css = f"""
            * {{
                font-family: sans-serif;
                font-size: {size}pt;
                margin: 0;
                padding: 0;
                line-height: 1.2;
            }}
            """
            
            # Try insert_htmlbox which handles wrapping
            inserted = False
            for scale in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
                scaled_size = size * scale
                scaled_css = css.replace(f'{size}pt', f'{scaled_size}pt')
                
                try:
                    page.insert_htmlbox(
                        rect,
                        html,
                        css=scaled_css
                    )
                    inserted = True
                    success_count += 1
                    self.stats['successful'] += 1
                    
                    if scale < 1.0 and idx < 3:
                        print(f"      ℹ Scaled to {scale*100:.0f}%: '{text[:40]}...'")
                    break
                    
                except Exception as e:
                    if scale == 0.5:  # Last attempt
                        self.stats['failed'] += 1
                        if self.stats['failed'] <= 3:
                            print(f"      ✗ Failed: '{text[:40]}...'")
                        self.stats['errors'].append(f"Page {page_num+1}, span {idx}: {str(e)}")
                    continue
        
        print(f"    ✓ Inserted: {success_count}/{len(translation_data)}")
        if success_count < len(translation_data):
            print(f"    ⚠ Failed: {len(translation_data) - success_count}")
    
    def process_all_pages(self, output_path: str = None):
        if output_path is None:
            base_name = os.path.splitext(self.original_pdf_path)[0]
            output_path = f"{base_name}_translated_ultimate.pdf"
        
        print(f"{'='*70}")
        print(f"ULTIMATE SURGICAL PDF REPLACEMENT")
        print(f"{'='*70}\n")
        
        for page_num in range(len(self.doc)):
            try:
                self.replace_page_text(page_num)
            except Exception as e:
                print(f"  ✗ Page {page_num + 1} error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*70}")
        print(f"Saving...")
        
        try:
            self.doc.save(output_path, garbage=4, deflate=True, clean=True)
            print(f"✓ Saved: {output_path}")
        except Exception as e:
            print(f"✗ Save error: {str(e)}")
            raise
        finally:
            self.doc.close()
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Total spans: {self.stats['total_spans']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {(self.stats['successful']/self.stats['total_spans']*100):.1f}%")
        
        if self.stats['errors'] and len(self.stats['errors']) <= 5:
            print(f"\nErrors:")
            for err in self.stats['errors']:
                print(f"  - {err}")
        
        print(f"{'='*70}\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: python surgical_replacement_ultimate.py <pdf> <translated_xml> [output]")
        sys.exit(1)
    
    original_pdf = sys.argv[1]
    translated_xml = sys.argv[2]
    output_pdf = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        replacer = UltimateSurgicalReplacer(original_pdf, translated_xml)
        replacer.process_all_pages(output_pdf)
    except Exception as e:
        print(f"\n✗ Fatal: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
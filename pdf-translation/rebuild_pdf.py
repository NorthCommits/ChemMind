"""
PDF Rebuild Script
Reconstructs a PDF from translated XML while preserving exact layout, positioning, and formatting.
"""

import xml.etree.ElementTree as ET
import os
import sys
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import Color
from PIL import Image


class PDFRebuilder:
    def __init__(self, xml_path: str, images_dir: str = "images"):
        """
        Initialize PDF Rebuilder.
        
        Args:
            xml_path: Path to translated XML file
            images_dir: Directory containing extracted images
        """
        self.xml_path = xml_path
        self.images_dir = images_dir
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        
        # Font mapping - map common PDF fonts to reportlab fonts
        self.font_mapping = {
            'Arial': 'Helvetica',
            'ArialMT': 'Helvetica',
            'Arial-BoldMT': 'Helvetica-Bold',
            'TimesNewRoman': 'Times-Roman',
            'TimesNewRomanPSMT': 'Times-Roman',
            'CourierNew': 'Courier',
        }
    
    def map_font(self, font_name: str) -> str:
        """
        Map PDF font names to ReportLab font names.
        
        Args:
            font_name: Original PDF font name
            
        Returns:
            ReportLab compatible font name
        """
        # Check if it's already a standard font
        standard_fonts = [
            'Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique', 'Helvetica-BoldOblique',
            'Times-Roman', 'Times-Bold', 'Times-Italic', 'Times-BoldItalic',
            'Courier', 'Courier-Bold', 'Courier-Oblique', 'Courier-BoldOblique'
        ]
        
        if font_name in standard_fonts:
            return font_name
        
        # Try to map to known fonts
        for key, value in self.font_mapping.items():
            if key.lower() in font_name.lower():
                return value
        
        # Check for bold/italic variants
        font_lower = font_name.lower()
        if 'bold' in font_lower and 'italic' in font_lower:
            return 'Helvetica-BoldOblique'
        elif 'bold' in font_lower:
            return 'Helvetica-Bold'
        elif 'italic' in font_lower or 'oblique' in font_lower:
            return 'Helvetica-Oblique'
        
        # Default to Helvetica
        return 'Helvetica'
    
    def get_color_from_int(self, color_int: int) -> Color:
        """
        Convert integer color value to ReportLab Color object.
        
        Args:
            color_int: Integer representation of color
            
        Returns:
            Color object
        """
        try:
            # Extract RGB components
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF
            return Color(r/255.0, g/255.0, b/255.0)
        except:
            return Color(0, 0, 0)  # Default to black
    
    def draw_text_span(self, c: canvas.Canvas, span: ET.Element, page_height: float):
        """
        Draw a text span on the canvas.
        
        Args:
            c: ReportLab canvas
            span: XML span element
            page_height: Height of the page for coordinate conversion
        """
        text = span.text
        if not text or not text.strip():
            return
        
        # Get positioning
        x0 = float(span.get('x0', 0))
        y0 = float(span.get('y0', 0))
        
        # Convert coordinates (PDF uses bottom-left origin, but we need to flip Y)
        y_converted = page_height - y0
        
        # Get font information
        font_name = span.get('font', 'Helvetica')
        font_size = float(span.get('size', 12))
        
        # Map font to ReportLab compatible font
        reportlab_font = self.map_font(font_name)
        
        try:
            c.setFont(reportlab_font, font_size)
        except:
            c.setFont('Helvetica', font_size)
        
        # Set text color if available
        color_val = span.get('color')
        if color_val:
            try:
                color = self.get_color_from_int(int(color_val))
                c.setFillColor(color)
            except:
                pass
        
        # Draw the text
        try:
            c.drawString(x0, y_converted, text)
        except Exception as e:
            print(f"Warning: Could not draw text '{text[:30]}...': {str(e)}")
    
    def draw_image(self, c: canvas.Canvas, img_elem: ET.Element, page_height: float):
        """
        Draw an image on the canvas.
        
        Args:
            c: ReportLab canvas
            img_elem: XML image element
            page_height: Height of the page for coordinate conversion
        """
        # Get image path
        image_path = img_elem.get('path')
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return
        
        # Get positioning
        x0 = float(img_elem.get('x0', 0))
        y0 = float(img_elem.get('y0', 0))
        x1 = float(img_elem.get('x1', 0))
        y1 = float(img_elem.get('y1', 0))
        
        # Calculate dimensions
        width = x1 - x0
        height = y1 - y0
        
        # Convert Y coordinate
        y_converted = page_height - y1
        
        try:
            # Draw the image
            c.drawImage(image_path, x0, y_converted, width=width, height=height, 
                       preserveAspectRatio=True, mask='auto')
        except Exception as e:
            print(f"Warning: Could not draw image {image_path}: {str(e)}")
    
    def draw_table(self, c: canvas.Canvas, table_elem: ET.Element, page_height: float):
        """
        Draw a table on the canvas.
        
        Args:
            c: ReportLab canvas
            table_elem: XML table element
            page_height: Height of the page for coordinate conversion
        """
        # Get table bounds
        x0 = float(table_elem.get('x0', 0))
        y0 = float(table_elem.get('y0', 0))
        x1 = float(table_elem.get('x1', 0))
        y1 = float(table_elem.get('y1', 0))
        
        rows = int(table_elem.get('rows', 0))
        cols = int(table_elem.get('cols', 0))
        
        if rows == 0 or cols == 0:
            return
        
        # Calculate cell dimensions
        table_width = x1 - x0
        table_height = y1 - y0
        cell_width = table_width / cols
        cell_height = table_height / rows
        
        # Convert Y coordinate
        y_converted = page_height - y1
        
        # Draw table grid
        c.setStrokeColor(Color(0, 0, 0))
        c.setLineWidth(0.5)
        
        # Draw horizontal lines
        for i in range(rows + 1):
            y = y_converted + (i * cell_height)
            c.line(x0, y, x1, y)
        
        # Draw vertical lines
        for i in range(cols + 1):
            x = x0 + (i * cell_width)
            c.line(x, y_converted, x, y_converted + table_height)
        
        # Draw cell contents
        cells = table_elem.findall('cell')
        c.setFont('Helvetica', 10)
        
        for cell in cells:
            row = int(cell.get('row', 0))
            col = int(cell.get('col', 0))
            text = cell.text
            
            if text:
                # Calculate cell position
                cell_x = x0 + (col * cell_width) + 2  # 2pt padding
                cell_y = y_converted + (row * cell_height) + cell_height - 12  # Position text
                
                try:
                    # Truncate text if too long
                    max_chars = int(cell_width / 6)  # Approximate chars that fit
                    if len(text) > max_chars:
                        text = text[:max_chars-3] + '...'
                    
                    c.drawString(cell_x, cell_y, text)
                except Exception as e:
                    print(f"Warning: Could not draw cell text: {str(e)}")
    
    def draw_drawing(self, c: canvas.Canvas, drawing_elem: ET.Element, page_height: float):
        """
        Draw vector graphics on the canvas.
        
        Args:
            c: ReportLab canvas
            drawing_elem: XML drawing element
            page_height: Height of the page for coordinate conversion
        """
        drawing_type = drawing_elem.get('type', 'unknown')
        
        # Get rectangle bounds if available
        x0 = drawing_elem.get('x0')
        y0 = drawing_elem.get('y0')
        x1 = drawing_elem.get('x1')
        y1 = drawing_elem.get('y1')
        
        if not all([x0, y0, x1, y1]):
            return
        
        x0 = float(x0)
        y0 = float(y0)
        x1 = float(x1)
        y1 = float(y1)
        
        # Convert Y coordinates
        y0_converted = page_height - y0
        y1_converted = page_height - y1
        
        # Draw based on type
        c.setStrokeColor(Color(0, 0, 0))
        c.setLineWidth(1)
        
        try:
            if 'rect' in drawing_type.lower() or 'line' in drawing_type.lower():
                # Draw as rectangle
                width = x1 - x0
                height = y0 - y1  # Note: y coordinates are flipped
                c.rect(x0, y1_converted, width, height, stroke=1, fill=0)
        except Exception as e:
            print(f"Warning: Could not draw shape: {str(e)}")
    
    def rebuild_pdf(self, output_path: str = None):
        """
        Rebuild the PDF from XML.
        
        Args:
            output_path: Output PDF file path
        """
        if output_path is None:
            base_name = os.path.splitext(self.xml_path)[0]
            output_path = f"{base_name}_rebuilt.pdf"
        
        print(f"Rebuilding PDF from: {self.xml_path}")
        print(f"Output: {output_path}")
        
        # Get document info
        source = self.root.get('source', 'Unknown')
        total_pages = int(self.root.get('pages', 0))
        
        print(f"Source: {source}")
        print(f"Total pages: {total_pages}")
        
        # Create PDF canvas (first page to get started)
        first_page = self.root.find('page')
        if first_page is None:
            print("Error: No pages found in XML!")
            return
        
        page_width = float(first_page.get('width', 612))
        page_height = float(first_page.get('height', 792))
        
        # Create canvas
        c = canvas.Canvas(output_path, pagesize=(page_width, page_height))
        
        # Process each page
        for page_num, page in enumerate(self.root.findall('page'), 1):
            print(f"\nProcessing page {page_num}/{total_pages}...")
            
            # Get page dimensions
            page_width = float(page.get('width', 612))
            page_height = float(page.get('height', 792))
            
            # Set page size if different
            c.setPageSize((page_width, page_height))
            
            # Draw text blocks
            text_blocks = page.find('text_blocks')
            if text_blocks is not None:
                spans = text_blocks.findall('.//span')
                print(f"  Drawing {len(spans)} text spans...")
                for span in spans:
                    self.draw_text_span(c, span, page_height)
            
            # Draw images
            images_section = page.find('images')
            if images_section is not None:
                images = images_section.findall('image')
                print(f"  Drawing {len(images)} images...")
                for img in images:
                    self.draw_image(c, img, page_height)
            
            # Draw tables
            tables_section = page.find('tables')
            if tables_section is not None:
                tables = tables_section.findall('table')
                print(f"  Drawing {len(tables)} tables...")
                for table in tables:
                    self.draw_table(c, table, page_height)
            
            # Draw drawings/shapes
            drawings_section = page.find('drawings')
            if drawings_section is not None:
                drawings = drawings_section.findall('drawing')
                if drawings:
                    print(f"  Drawing {len(drawings)} vector graphics...")
                    for drawing in drawings:
                        self.draw_drawing(c, drawing, page_height)
            
            # Show page (move to next)
            c.showPage()
        
        # Save the PDF
        c.save()
        
        print(f"\n✅ PDF rebuilt successfully!")
        print(f"Output file: {output_path}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python rebuild_pdf.py <translated_xml> [output_pdf] [images_dir]")
        print("\nExamples:")
        print("  python rebuild_pdf.py document_translated_spanish.xml")
        print("  python rebuild_pdf.py document_translated_spanish.xml output.pdf")
        print("  python rebuild_pdf.py document_translated_spanish.xml output.pdf images")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    images_dir = sys.argv[3] if len(sys.argv) > 3 else "images"
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file '{xml_path}' not found!")
        sys.exit(1)
    
    # Create rebuilder and rebuild PDF
    rebuilder = PDFRebuilder(xml_path, images_dir)
    rebuilder.rebuild_pdf(output_pdf)


if __name__ == "__main__":
    main()
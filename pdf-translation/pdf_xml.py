"""
PDF to XML Converter
Converts PDF files to XML format, extracting text, images, tables, drawings, and their coordinates.
"""

import pymupdf  # PyMuPDF
import base64
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import sys


def extract_images(page, page_num, output_dir="images"):
    """Extract images from a PDF page and return their information."""
    images_data = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_list = page.get_images()
    
    for img_index, img in enumerate(image_list):
        try:
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image to file
            image_filename = f"page_{page_num}_image_{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            # Try to get image position on page
            # The img tuple contains: (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, bbox)
            image_bbox = None
            try:
                # Try method 1: get_image_bbox
                image_bbox = page.get_image_bbox(img)
            except (ValueError, Exception):
                # Method 2: Use bbox from image list if available (index 9)
                if len(img) > 9 and img[9]:
                    image_bbox = img[9]
                else:
                    # Method 3: Search for image in page items
                    try:
                        img_dict = page.get_text("dict")
                        for block in img_dict.get("blocks", []):
                            if block.get("type") == 1:  # Image block
                                if block.get("image") == xref or block.get("xref") == xref:
                                    image_bbox = block.get("bbox")
                                    break
                    except Exception:
                        pass
            
            images_data.append({
                "filename": image_filename,
                "path": image_path,
                "extension": image_ext,
                "xref": xref,
                "bbox": image_bbox,
                "width": base_image.get("width", 0),
                "height": base_image.get("height", 0)
            })
        
        except Exception as e:
            print(f"Warning: Could not extract image {img_index} on page {page_num}: {str(e)}")
            continue
    
    return images_data


def extract_tables(page):
    """Extract tables from a PDF page."""
    tables_data = []
    
    try:
        tabs = page.find_tables()
        
        for table_index, table in enumerate(tabs):
            try:
                table_info = {
                    "index": table_index,
                    "bbox": table.bbox,
                    "rows": table.row_count,
                    "cols": table.col_count,
                    "cells": []
                }
                
                # Extract table data
                table_data = table.extract()
                
                for row_idx, row in enumerate(table_data):
                    for col_idx, cell in enumerate(row):
                        if cell:  # Only add non-empty cells
                            table_info["cells"].append({
                                "row": row_idx,
                                "col": col_idx,
                                "content": str(cell)
                            })
                
                tables_data.append(table_info)
            except Exception as e:
                print(f"Warning: Could not extract table {table_index}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Warning: Could not process tables: {str(e)}")
    
    return tables_data


def extract_drawings(page):
    """Extract vector graphics/drawings from a PDF page."""
    drawings_data = []
    
    try:
        drawings = page.get_drawings()
        
        for draw_index, drawing in enumerate(drawings):
            try:
                drawing_info = {
                    "index": draw_index,
                    "type": drawing.get("type", "unknown"),
                    "rect": drawing.get("rect"),
                    "color": drawing.get("color"),
                    "fill": drawing.get("fill"),
                    "width": drawing.get("width"),
                    "items": []
                }
                
                # Extract path items (lines, curves, etc.)
                if "items" in drawing:
                    for item in drawing["items"]:
                        drawing_info["items"].append({
                            "type": item[0] if isinstance(item, tuple) else "unknown",
                            "points": str(item[1:]) if len(item) > 1 else ""
                        })
                
                drawings_data.append(drawing_info)
            except Exception as e:
                print(f"Warning: Could not extract drawing {draw_index}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Warning: Could not process drawings: {str(e)}")
    
    return drawings_data


def extract_text_blocks(page):
    """Extract text blocks with positioning information."""
    text_blocks = []
    
    try:
        # Get text as dictionary with detailed information
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            try:
                if block.get("type") == 0:  # Text block
                    block_info = {
                        "bbox": block.get("bbox"),
                        "lines": []
                    }
                    
                    for line in block.get("lines", []):
                        line_info = {
                            "bbox": line.get("bbox"),
                            "spans": []
                        }
                        
                        for span in line.get("spans", []):
                            line_info["spans"].append({
                                "text": span.get("text", ""),
                                "bbox": span.get("bbox"),
                                "font": span.get("font", ""),
                                "size": span.get("size", 0),
                                "color": span.get("color", 0)
                            })
                        
                        block_info["lines"].append(line_info)
                    
                    text_blocks.append(block_info)
            except Exception as e:
                print(f"Warning: Could not extract text block: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Warning: Could not process text blocks: {str(e)}")
    
    return text_blocks


def create_xml_structure(pdf_path):
    """Create XML structure from PDF content."""
    doc = pymupdf.open(pdf_path)
    
    # Create root element
    root = ET.Element("document")
    root.set("source", os.path.basename(pdf_path))
    root.set("pages", str(len(doc)))
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        page_elem = ET.SubElement(root, "page")
        page_elem.set("number", str(page_num + 1))
        page_elem.set("width", str(page.rect.width))
        page_elem.set("height", str(page.rect.height))
        
        # Extract and add text blocks
        text_blocks = extract_text_blocks(page)
        if text_blocks:
            text_section = ET.SubElement(page_elem, "text_blocks")
            for block_idx, block in enumerate(text_blocks):
                block_elem = ET.SubElement(text_section, "block")
                block_elem.set("id", str(block_idx))
                if block["bbox"]:
                    block_elem.set("x0", str(block["bbox"][0]))
                    block_elem.set("y0", str(block["bbox"][1]))
                    block_elem.set("x1", str(block["bbox"][2]))
                    block_elem.set("y1", str(block["bbox"][3]))
                
                for line_idx, line in enumerate(block["lines"]):
                    line_elem = ET.SubElement(block_elem, "line")
                    line_elem.set("id", str(line_idx))
                    if line["bbox"]:
                        line_elem.set("x0", str(line["bbox"][0]))
                        line_elem.set("y0", str(line["bbox"][1]))
                        line_elem.set("x1", str(line["bbox"][2]))
                        line_elem.set("y1", str(line["bbox"][3]))
                    
                    for span in line["spans"]:
                        span_elem = ET.SubElement(line_elem, "span")
                        span_elem.set("font", span["font"])
                        span_elem.set("size", str(span["size"]))
                        span_elem.text = span["text"]
                        if span["bbox"]:
                            span_elem.set("x0", str(span["bbox"][0]))
                            span_elem.set("y0", str(span["bbox"][1]))
                            span_elem.set("x1", str(span["bbox"][2]))
                            span_elem.set("y1", str(span["bbox"][3]))
        
        # Extract and add images
        images = extract_images(page, page_num + 1)
        if images:
            images_section = ET.SubElement(page_elem, "images")
            for img in images:
                img_elem = ET.SubElement(images_section, "image")
                img_elem.set("filename", img["filename"])
                img_elem.set("path", img["path"])
                img_elem.set("extension", img["extension"])
                img_elem.set("width", str(img["width"]))
                img_elem.set("height", str(img["height"]))
                if img["bbox"]:
                    img_elem.set("x0", str(img["bbox"][0]))
                    img_elem.set("y0", str(img["bbox"][1]))
                    img_elem.set("x1", str(img["bbox"][2]))
                    img_elem.set("y1", str(img["bbox"][3]))
        
        # Extract and add tables
        tables = extract_tables(page)
        if tables:
            tables_section = ET.SubElement(page_elem, "tables")
            for table in tables:
                table_elem = ET.SubElement(tables_section, "table")
                table_elem.set("id", str(table["index"]))
                table_elem.set("rows", str(table["rows"]))
                table_elem.set("cols", str(table["cols"]))
                if table["bbox"]:
                    table_elem.set("x0", str(table["bbox"][0]))
                    table_elem.set("y0", str(table["bbox"][1]))
                    table_elem.set("x1", str(table["bbox"][2]))
                    table_elem.set("y1", str(table["bbox"][3]))
                
                for cell in table["cells"]:
                    cell_elem = ET.SubElement(table_elem, "cell")
                    cell_elem.set("row", str(cell["row"]))
                    cell_elem.set("col", str(cell["col"]))
                    cell_elem.text = cell["content"]
        
        # Extract and add drawings/vector graphics
        drawings = extract_drawings(page)
        if drawings:
            drawings_section = ET.SubElement(page_elem, "drawings")
            for drawing in drawings:
                draw_elem = ET.SubElement(drawings_section, "drawing")
                draw_elem.set("id", str(drawing["index"]))
                draw_elem.set("type", drawing["type"])
                if drawing["rect"]:
                    draw_elem.set("x0", str(drawing["rect"][0]))
                    draw_elem.set("y0", str(drawing["rect"][1]))
                    draw_elem.set("x1", str(drawing["rect"][2]))
                    draw_elem.set("y1", str(drawing["rect"][3]))
                
                if drawing["items"]:
                    items_elem = ET.SubElement(draw_elem, "items")
                    for item in drawing["items"]:
                        item_elem = ET.SubElement(items_elem, "item")
                        item_elem.set("type", str(item["type"]))
                        item_elem.text = item["points"]
    
    doc.close()
    return root


def prettify_xml(elem):
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def convert_pdf_to_xml(pdf_path, output_xml_path=None):
    """
    Main function to convert PDF to XML.
    
    Args:
        pdf_path: Path to input PDF file
        output_xml_path: Path to output XML file (optional)
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found!")
        return
    
    print(f"Processing PDF: {pdf_path}")
    
    # Create XML structure
    xml_root = create_xml_structure(pdf_path)
    
    # Generate output filename if not provided
    if output_xml_path is None:
        output_xml_path = os.path.splitext(pdf_path)[0] + ".xml"
    
    # Write to file
    xml_string = prettify_xml(xml_root)
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_string)
    
    print(f"XML file created: {output_xml_path}")
    print("Images saved to: ./images/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_xml_converter.py <pdf_file> [output_xml_file]")
        print("Example: python pdf_to_xml_converter.py document.pdf output.xml")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_pdf_to_xml(pdf_file, output_file)
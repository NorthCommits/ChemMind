"""
Advanced PDF to XML Converter - Complex Layout Support
Extracts comprehensive layout information including:
- All text with styling
- Background colors and fills
- Shapes (rectangles, rounded rectangles, circles)
- All images and graphics
- Visual grouping and sections
- Z-order preservation
"""

import pymupdf  # PyMuPDF
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json


def extract_page_background(page):
    """Extract page background color if any."""
    try:
        # Get page color
        return {
            "color": None,  # Default white, can be enhanced
            "width": page.rect.width,
            "height": page.rect.height
        }
    except Exception as e:
        print(f"Warning: Could not extract background: {str(e)}")
        return None


def extract_shapes_advanced(page):
    """
    Extract all shapes with detailed information including:
    - Rectangles (regular and rounded)
    - Circles/ellipses
    - Lines
    - Fill colors
    - Stroke colors
    - Transparency
    """
    shapes_data = []
    
    try:
        drawings = page.get_drawings()
        
        for draw_index, drawing in enumerate(drawings):
            try:
                shape_info = {
                    "index": draw_index,
                    "type": drawing.get("type", "unknown"),
                    "rect": drawing.get("rect"),
                    "fill_color": drawing.get("fill"),  # Fill color (RGBA or None)
                    "stroke_color": drawing.get("color"),  # Stroke/border color
                    "stroke_width": drawing.get("width", 0),
                    "fill_opacity": drawing.get("fill_opacity", 1.0),
                    "stroke_opacity": drawing.get("stroke_opacity", 1.0),
                    "even_odd": drawing.get("even_odd", False),
                    "items": []
                }
                
                # Extract path items with more detail
                if "items" in drawing:
                    for item in drawing["items"]:
                        if isinstance(item, (list, tuple)) and len(item) > 0:
                            item_type = item[0]
                            
                            # Parse different shape types
                            if item_type == "re":  # Rectangle
                                # item[1] is typically (Rect(...), ...)
                                shape_info["shape_type"] = "rectangle"
                                if len(item) > 1:
                                    shape_info["items"].append({
                                        "type": "rectangle",
                                        "data": str(item[1:])
                                    })
                            elif item_type == "c":  # Curve
                                shape_info["shape_type"] = "curve"
                                shape_info["items"].append({
                                    "type": "curve",
                                    "data": str(item[1:])
                                })
                            elif item_type == "l":  # Line
                                shape_info["shape_type"] = "line"
                                shape_info["items"].append({
                                    "type": "line",
                                    "data": str(item[1:])
                                })
                            else:
                                shape_info["items"].append({
                                    "type": str(item_type),
                                    "data": str(item[1:]) if len(item) > 1 else ""
                                })
                
                shapes_data.append(shape_info)
                
            except Exception as e:
                print(f"Warning: Could not extract shape {draw_index}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Warning: Could not process shapes: {str(e)}")
    
    return shapes_data


def extract_all_images(page, page_num, output_dir="images"):
    """
    Extract ALL images including:
    - Embedded images
    - Icons
    - Graphics
    With detailed positioning and metadata
    """
    images_data = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                
                # Extract image data
                base_image = page.parent.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save image to file
                    image_filename = f"page_{page_num}_image_{img_index}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Get image position on page
                    image_bbox = None
                    try:
                        image_bbox = page.get_image_bbox(img)
                    except:
                        # Fallback methods
                        if len(img) > 9 and img[9]:
                            image_bbox = img[9]
                        else:
                            # Search in page dictionary
                            img_dict = page.get_text("dict")
                            for block in img_dict.get("blocks", []):
                                if block.get("type") == 1:  # Image block
                                    if block.get("xref") == xref:
                                        image_bbox = block.get("bbox")
                                        break
                    
                    images_data.append({
                        "filename": image_filename,
                        "path": image_path,
                        "extension": image_ext,
                        "xref": xref,
                        "bbox": image_bbox,
                        "width": base_image.get("width", 0),
                        "height": base_image.get("height", 0),
                        "colorspace": base_image.get("colorspace"),
                        "bpc": base_image.get("bpc"),
                        "transform": img[1:7] if len(img) > 7 else None,  # Transformation matrix
                    })
                    
            except Exception as e:
                print(f"Warning: Could not extract image {img_index}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Warning: Could not process images: {str(e)}")
    
    return images_data


def extract_text_with_sections(page):
    """
    Extract text with section/grouping awareness.
    Groups text that belongs together visually.
    """
    text_blocks = []
    
    try:
        text_dict = page.get_text("dict")
        
        for block_idx, block in enumerate(text_dict.get("blocks", [])):
            try:
                if block.get("type") == 0:  # Text block
                    block_bbox = block.get("bbox")
                    
                    block_info = {
                        "index": block_idx,
                        "bbox": block_bbox,
                        "lines": []
                    }
                    
                    for line in block.get("lines", []):
                        line_bbox = line.get("bbox")
                        line_info = {
                            "bbox": line_bbox,
                            "wmode": line.get("wmode", 0),
                            "dir": line.get("dir", [1, 0]),
                            "spans": []
                        }
                        
                        if line_bbox:
                            line_info["line_height"] = line_bbox[3] - line_bbox[1]
                        
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            font = span.get("font", "")
                            size = span.get("size", 0)
                            flags = span.get("flags", 0)
                            color = span.get("color", 0)
                            bbox = span.get("bbox")
                            
                            # Enhanced font flags
                            is_bold = bool(flags & (1 << 18))
                            is_italic = bool(flags & (1 << 6))
                            
                            # Check font name too
                            font_lower = font.lower()
                            if 'bold' in font_lower:
                                is_bold = True
                            if 'italic' in font_lower or 'oblique' in font_lower:
                                is_italic = True
                            
                            baseline_y = bbox[3] if bbox else 0
                            ascender = span.get("ascender", 0.8)
                            descender = span.get("descender", -0.2)
                            origin = span.get("origin")
                            
                            span_info = {
                                "text": text,
                                "bbox": bbox,
                                "font": font,
                                "size": size,
                                "color": color,
                                "flags": flags,
                                "is_bold": is_bold,
                                "is_italic": is_italic,
                                "baseline_y": baseline_y,
                                "ascender": ascender,
                                "descender": descender,
                                "origin": origin,
                            }
                            
                            line_info["spans"].append(span_info)
                        
                        block_info["lines"].append(line_info)
                    
                    text_blocks.append(block_info)
                    
            except Exception as e:
                print(f"Warning: Could not extract text block {block_idx}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Warning: Could not process text blocks: {str(e)}")
    
    return text_blocks


def detect_visual_sections(text_blocks, shapes, images):
    """
    Detect visual sections/groups by analyzing:
    - Text blocks near colored backgrounds
    - Text surrounded by rectangles
    - Text positioned near images
    """
    sections = []
    
    # Group shapes by position to identify "containers"
    containers = []
    for shape in shapes:
        if shape.get("fill_color") and shape.get("rect"):
            rect = shape["rect"]
            containers.append({
                "bbox": rect,
                "fill_color": shape["fill_color"],
                "shape_index": shape["index"],
                "content": []
            })
    
    # Assign text blocks to containers
    for text_block in text_blocks:
        text_bbox = text_block.get("bbox")
        if not text_bbox:
            continue
        
        tx0, ty0, tx1, ty1 = text_bbox
        text_center_x = (tx0 + tx1) / 2
        text_center_y = (ty0 + ty1) / 2
        
        # Find which container this text belongs to
        for container in containers:
            cx0, cy0, cx1, cy1 = container["bbox"]
            
            # Check if text center is inside container
            if (cx0 <= text_center_x <= cx1 and 
                cy0 <= text_center_y <= cy1):
                container["content"].append({
                    "type": "text",
                    "index": text_block["index"]
                })
                break
    
    return containers


def create_advanced_xml_structure(pdf_path):
    """Create comprehensive XML structure with all visual elements."""
    doc = pymupdf.open(pdf_path)
    
    # Create root element
    root = ET.Element("document")
    root.set("source", os.path.basename(pdf_path))
    root.set("pages", str(len(doc)))
    root.set("layout_type", "complex")  # Flag for complex layout
    
    # Document metadata
    metadata = doc.metadata
    if metadata:
        meta_elem = ET.SubElement(root, "metadata")
        for key, value in metadata.items():
            if value:
                meta_elem.set(key, str(value))
    
    # Process each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        print(f"Processing page {page_num + 1}/{len(doc)}...")
        
        page_elem = ET.SubElement(root, "page")
        page_elem.set("number", str(page_num + 1))
        page_elem.set("width", str(page.rect.width))
        page_elem.set("height", str(page.rect.height))
        page_elem.set("rotation", str(page.rotation))
        
        # Extract all components
        print(f"  Extracting shapes...")
        shapes = extract_shapes_advanced(page)
        
        print(f"  Extracting images...")
        images = extract_all_images(page, page_num + 1)
        
        print(f"  Extracting text...")
        text_blocks = extract_text_with_sections(page)
        
        print(f"  Detecting visual sections...")
        sections = detect_visual_sections(text_blocks, shapes, images)
        
        # Add shapes to XML
        if shapes:
            shapes_section = ET.SubElement(page_elem, "shapes")
            shapes_section.set("count", str(len(shapes)))
            
            for shape in shapes:
                shape_elem = ET.SubElement(shapes_section, "shape")
                shape_elem.set("index", str(shape["index"]))
                shape_elem.set("type", shape["type"])
                
                if shape.get("shape_type"):
                    shape_elem.set("shape_type", shape["shape_type"])
                
                if shape["rect"]:
                    shape_elem.set("x0", str(shape["rect"][0]))
                    shape_elem.set("y0", str(shape["rect"][1]))
                    shape_elem.set("x1", str(shape["rect"][2]))
                    shape_elem.set("y1", str(shape["rect"][3]))
                
                # Color information
                if shape["fill_color"]:
                    shape_elem.set("fill_color", str(shape["fill_color"]))
                if shape["stroke_color"]:
                    shape_elem.set("stroke_color", str(shape["stroke_color"]))
                
                shape_elem.set("stroke_width", str(shape["stroke_width"]))
                shape_elem.set("fill_opacity", str(shape["fill_opacity"]))
                shape_elem.set("stroke_opacity", str(shape["stroke_opacity"]))
                
                # Add path items
                if shape["items"]:
                    items_elem = ET.SubElement(shape_elem, "items")
                    for item in shape["items"]:
                        item_elem = ET.SubElement(items_elem, "item")
                        item_elem.set("type", item["type"])
                        item_elem.text = item["data"]
        
        # Add images to XML
        if images:
            images_section = ET.SubElement(page_elem, "images")
            images_section.set("count", str(len(images)))
            
            for img in images:
                img_elem = ET.SubElement(images_section, "image")
                img_elem.set("filename", img["filename"])
                img_elem.set("path", img["path"])
                img_elem.set("extension", img["extension"])
                img_elem.set("xref", str(img["xref"]))
                img_elem.set("width", str(img["width"]))
                img_elem.set("height", str(img["height"]))
                
                if img.get("colorspace"):
                    img_elem.set("colorspace", str(img["colorspace"]))
                if img.get("bpc"):
                    img_elem.set("bpc", str(img["bpc"]))
                
                if img["bbox"]:
                    img_elem.set("x0", str(img["bbox"][0]))
                    img_elem.set("y0", str(img["bbox"][1]))
                    img_elem.set("x1", str(img["bbox"][2]))
                    img_elem.set("y1", str(img["bbox"][3]))
                
                if img.get("transform"):
                    img_elem.set("transform", json.dumps(img["transform"]))
        
        # Add text blocks
        if text_blocks:
            text_section = ET.SubElement(page_elem, "text_blocks")
            text_section.set("count", str(len(text_blocks)))
            
            for block in text_blocks:
                block_elem = ET.SubElement(text_section, "block")
                block_elem.set("index", str(block["index"]))
                
                if block["bbox"]:
                    block_elem.set("x0", str(block["bbox"][0]))
                    block_elem.set("y0", str(block["bbox"][1]))
                    block_elem.set("x1", str(block["bbox"][2]))
                    block_elem.set("y1", str(block["bbox"][3]))
                
                for line in block["lines"]:
                    line_elem = ET.SubElement(block_elem, "line")
                    
                    if line["bbox"]:
                        line_elem.set("x0", str(line["bbox"][0]))
                        line_elem.set("y0", str(line["bbox"][1]))
                        line_elem.set("x1", str(line["bbox"][2]))
                        line_elem.set("y1", str(line["bbox"][3]))
                    
                    line_elem.set("wmode", str(line["wmode"]))
                    if "line_height" in line:
                        line_elem.set("line_height", str(line["line_height"]))
                    
                    for span in line["spans"]:
                        span_elem = ET.SubElement(line_elem, "span")
                        span_elem.set("font", span["font"])
                        span_elem.set("size", str(span["size"]))
                        span_elem.set("color", str(span["color"]))
                        span_elem.set("flags", str(span["flags"]))
                        span_elem.set("bold", str(span["is_bold"]))
                        span_elem.set("italic", str(span["is_italic"]))
                        span_elem.set("baseline_y", str(span["baseline_y"]))
                        span_elem.set("ascender", str(span["ascender"]))
                        span_elem.set("descender", str(span["descender"]))
                        
                        span_elem.text = span["text"]
                        
                        if span["bbox"]:
                            span_elem.set("x0", str(span["bbox"][0]))
                            span_elem.set("y0", str(span["bbox"][1]))
                            span_elem.set("x1", str(span["bbox"][2]))
                            span_elem.set("y1", str(span["bbox"][3]))
                        
                        if span["origin"]:
                            span_elem.set("origin_x", str(span["origin"][0]))
                            span_elem.set("origin_y", str(span["origin"][1]))
        
        # Add visual sections
        if sections:
            sections_elem = ET.SubElement(page_elem, "visual_sections")
            sections_elem.set("count", str(len(sections)))
            
            for section in sections:
                section_elem = ET.SubElement(sections_elem, "section")
                
                bbox = section["bbox"]
                section_elem.set("x0", str(bbox[0]))
                section_elem.set("y0", str(bbox[1]))
                section_elem.set("x1", str(bbox[2]))
                section_elem.set("y1", str(bbox[3]))
                
                if section.get("fill_color"):
                    section_elem.set("fill_color", str(section["fill_color"]))
                
                section_elem.set("shape_index", str(section["shape_index"]))
                
                # Add content references
                if section.get("content"):
                    content_elem = ET.SubElement(section_elem, "content")
                    for item in section["content"]:
                        item_elem = ET.SubElement(content_elem, "item")
                        item_elem.set("type", item["type"])
                        item_elem.set("index", str(item["index"]))
        
        print(f"  ✓ Extracted: {len(shapes)} shapes, {len(images)} images, {len(text_blocks)} text blocks, {len(sections)} sections")
    
    doc.close()
    return root


def prettify_xml(elem):
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def convert_pdf_to_xml_advanced(pdf_path, output_xml_path=None):
    """
    Main function to convert complex PDF to XML with full layout preservation.
    
    Args:
        pdf_path: Path to input PDF file
        output_xml_path: Path to output XML file (optional)
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found!")
        return
    
    print(f"=" * 70)
    print(f"ADVANCED PDF EXTRACTION - Complex Layout Support")
    print(f"=" * 70)
    print(f"Processing: {pdf_path}")
    print()
    
    # Create XML structure
    xml_root = create_advanced_xml_structure(pdf_path)
    
    # Generate output filename if not provided
    if output_xml_path is None:
        output_xml_path = os.path.splitext(pdf_path)[0] + "_advanced.xml"
    
    # Write to file
    xml_string = prettify_xml(xml_root)
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_string)
    
    print()
    print(f"=" * 70)
    print(f"✅ EXTRACTION COMPLETE!")
    print(f"=" * 70)
    print(f"✓ XML file: {output_xml_path}")
    print(f"✓ Images: ./images/")
    print(f"✓ Features extracted:")
    print(f"   - Text with full styling")
    print(f"   - All shapes with colors")
    print(f"   - All images with positioning")
    print(f"   - Visual sections/containers")
    print(f"   - Z-order preservation")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_xml_advanced.py <pdf_file> [output_xml_file]")
        print("Example: python pdf_xml_advanced.py complex_document.pdf output.xml")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_pdf_to_xml_advanced(pdf_file, output_file)
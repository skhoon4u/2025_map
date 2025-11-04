#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Road Detection Pipeline: Complete End-to-End System
From Map Image → GPS Coordinates of Road Highlights

Pipeline Stages:
  Stage 1: OCR → POI Extraction → Bounding Box from VWorld
    - Phase 1: Initial OCR + Crop OCR + Rule Filter + LLM Filter + Ranking + DB Search
    - Phase 2: All-in-One LLM Revision
    - Phase 3: Individual Crop-level LLM Revision
    - Phase 4: Bounding Box Extraction
  
  Stage 2: Map Alignment
    - Option A: Affine Transformation (using POI correspondences)
    - Option B: Feature Matching (sliding window similarity search)
  
  Stage 3: Highlight Extraction
    - DeepLabV3-based segmentation
    - Morphological processing
    - Skeletonization
  
  Stage 4: Highlight Mapping to VWorld
    - Transform extracted highlight to VWorld coordinates
    - Visualization on VWorld map
  
  Stage 5: GPS Coordinate Calculation
    - Convert pixel coordinates to GPS
    - Generate polyline
    - Export GeoJSON

Key Features:
- Complete automation from image to GPS coordinates
- Multi-stage error correction with LLM
- Advanced feature-based map alignment
- Professional output organization
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
from itertools import combinations
import re
import unicodedata
from scipy.ndimage import distance_transform_edt, gaussian_filter
import shutil

# Auto-detect project directory
if os.environ.get('PROJECT_DIR'):
    PROJECT_DIR = Path(os.environ['PROJECT_DIR'])
else:
    PROJECT_DIR = Path(__file__).parent.absolute()

sys.path.insert(0, str(PROJECT_DIR))

# Set CUDA device to 0 as requested
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Import after CUDA setup
import torch

# Import project modules
from paddleocr import PaddleOCR
from pyproj import CRS, Transformer
import requests
from geopy.distance import great_circle

# Highlight extraction utils
from skimage.morphology import medial_axis, skeletonize
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

class TeeLogger:
    """Captures print output to both console and log file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = log_file
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

# ============================================================================
# TEXT NORMALIZATION UTILITIES
# ============================================================================

def normalize_text_for_search(text: str) -> str:
    """
    Normalize text for consistent searching
    - Removes all whitespace
    - Converts to lowercase
    - Removes special punctuation (keeps Korean, alphanumeric, and common separators)
    - Normalizes Unicode
    """
    if not text:
        return ""
    
    # Normalize Unicode (NFC form)
    text = unicodedata.normalize('NFC', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove all whitespace
    text = ''.join(text.split())
    
    # Remove common punctuation but keep hyphen, parentheses for place names
    # Keep: Korean characters, alphanumeric, hyphen, parentheses
    allowed_chars = []
    for char in text:
        if (char.isalnum() or 
            '\uAC00' <= char <= '\uD7A3' or  # Korean Hangul
            char in '-()'):
            allowed_chars.append(char)
    
    text = ''.join(allowed_chars)
    
    return text


def normalize_text_simple(text: str) -> str:
    """
    Simple normalization for display
    - Strips leading/trailing whitespace
    - Normalizes Unicode
    """
    if not text:
        return ""
    
    # Normalize Unicode (NFC form)
    text = unicodedata.normalize('NFC', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def get_korean_font(font_size=22):
    """Get Korean font for visualization"""
    font_paths = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        'C:/Windows/Fonts/malgun.ttf',
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                pass
    
    return ImageFont.load_default()


def save_ocr_visualization(image_path: Path, ocr_data: Dict, output_path: Path, padding: int = 50):
    """Save OCR visualization with bounding boxes and labels"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ✗ Could not load image for visualization")
        return
    
    # Add padding
    img_padded = cv2.copyMakeBorder(
        img, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    
    # Convert to PIL for better text rendering
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_korean_font(16)
    
    # Draw each detection
    for entry in ocr_data.get('extracted_texts', []):
        bbox = entry.get('bounding_box', {})
        text = entry.get('name', '')
        
        # Get polygon points (adjusted for padding)
        tl = bbox.get('top_left', {})
        tr = bbox.get('top_right', {})
        br = bbox.get('bottom_right', {})
        bl = bbox.get('bottom_left', {})
        
        points = [
            (tl['x'] + padding, tl['y'] + padding),
            (tr['x'] + padding, tr['y'] + padding),
            (br['x'] + padding, br['y'] + padding),
            (bl['x'] + padding, bl['y'] + padding)
        ]
        
        # Draw polygon (red)
        draw.polygon(points, outline=(255, 0, 0), width=2)
        
        # Draw text label
        label = f"{entry['index']}: {text[:50]}" if len(text) > 50 else f"{entry['index']}: {text}"
        text_pos = (int(tl['x'] + padding), max(15, int(tl['y'] + padding) - 10))
        draw.text(text_pos, label, font=font, fill=(255, 0, 0))
    
    # Convert back to OpenCV format and save
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_cv)
    print(f"  ✓ Saved visualization: {output_path}")


def save_marked_vworld_image(vworld_image_path: Path, poi_pixels: List[Dict], 
                             output_path: Path):
    """Save VWorld image with POI markers"""
    img = cv2.imread(str(vworld_image_path))
    if img is None:
        print(f"  ✗ Could not load VWorld image for marking")
        return
    
    # Convert to PIL for proper Korean text rendering
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_korean_font(14)
    
    # Draw POI markers
    for poi in poi_pixels:
        x, y = int(poi['x']), int(poi['y'])
        name = poi['name']
        
        # Draw circle
        draw.ellipse([x-10, y-10, x+10, y+10], outline=(0, 255, 0), width=3)
        
        # Draw cross
        draw.line([x-10, y, x+10, y], fill=(0, 255, 0), width=3)
        draw.line([x, y-10, x, y+10], fill=(0, 255, 0), width=3)
        
        # Draw text with background
        bbox = draw.textbbox((x, y-8), name, font=font)
        draw.rectangle([bbox[0]-3, bbox[1]-2, bbox[2]+3, bbox[3]+2], fill=(0, 255, 0))
        draw.text((x, y-8), name, font=font, fill=(0, 0, 0))
    
    # Convert back to OpenCV
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_cv)
    print(f"  ✓ Saved marked VWorld image: {output_path}")


# ============================================================================
# LLM WRAPPER (Qwen3-VL + GPT)
# ============================================================================

class LLMWrapper:
    """LLM wrapper for vision-language tasks"""
    
    def __init__(self, model_type="qwen", reasoning_effort="minimal", **kwargs):
        self.model_type = model_type
        self.reasoning_effort = reasoning_effort
        self.model = None
        self.processor = None
        
        if model_type == "qwen":
            print(f"  Initializing Qwen3-VL (Qwen/Qwen3-VL-30B-A3B-Instruct)...")
            self._init_qwen()
        elif model_type == "gpt":
            self.api_key = kwargs.get("api_key")
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("GPT mode requires api_key or OPENAI_API_KEY")
            print(f"  Using GPT with reasoning capacity: {self.reasoning_effort}")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _init_qwen(self):
        """Initialize Qwen3-VL model following official quickstart"""
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
            
            # Set CUDA memory management
            if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            model_path = "Qwen/Qwen3-VL-30B-A3B-Instruct"
            
            # Load model with auto dtype and device map
            print(f"    Loading model (this may take a few minutes)...")
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
            )
            
            # Load processor
            print(f"    Loading processor...")
            self.processor = AutoProcessor.from_pretrained(model_path)
            
            print(f"    ✓ Qwen3-VL initialized successfully!")
            
        except ImportError as e:
            print(f"    ✗ Error: transformers library issue")
            print(f"    Please install latest transformers:")
            print(f"    pip install git+https://github.com/huggingface/transformers")
            raise
        except Exception as e:
            print(f"    ✗ Error initializing Qwen3-VL: {e}")
            raise
    
    def inference(self, image_path: str, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """
        Run VLM inference
        
        Args:
            image_path: Path to image file (can be None for text-only)
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model response as string, or None if failed
        """
        if self.model_type == "qwen":
            return self._qwen_inference(image_path, prompt, max_tokens)
        elif self.model_type == "gpt":
            return self._gpt_inference(image_path, prompt, max_tokens)
        else:
            return None
    
    def _qwen_inference(self, image_path: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Run Qwen3-VL inference following official pattern"""
        try:
            # Prepare messages
            content = []
            if image_path:
                content.append({"type": "image", "image": image_path})
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            # Prepare inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
            
            # Trim prompt tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # Decode
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else None
            
        except Exception as e:
            print(f"      Qwen inference error: {e}")
            return None
    
    def _gpt_inference(self, image_path: str, prompt: str, max_tokens: int) -> Optional[str]:
        """Run GPT inference using Responses API"""
        try:
            from openai import OpenAI
            import base64
            
            # Initialize client
            client = OpenAI(api_key=self.api_key)
            
            content = [{"type": "input_text", "text": prompt}]
            
            # Add image if provided
            if image_path:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                ext = Path(image_path).suffix.lower()
                media_type = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.webp': 'image/webp'
                }.get(ext, 'image/png')
                
                content.append({
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{base64_image}"
                })
            
            # Call GPT using Responses API with reasoning capacity
            response = client.responses.create(
                model="gpt-5",
                input=[{"role": "user", "content": content}],
                reasoning={"effort": self.reasoning_effort},
                max_output_tokens=max_tokens
            )
            
            return response.output_text
            
        except ImportError:
            print("      Error: openai package not installed")
            return None
        except Exception as e:
            print(f"      GPT error: {e}")
            return None


def extract_json_from_text(text: str, debug: bool = False) -> Optional[Dict]:
    """
    Robust JSON extraction from LLM response text
    """
    if not text:
        return None
    
    if debug:
        print(f"      [JSON Parse] Input length: {len(text)} chars")
    
    # Strategy 1: Code blocks
    code_block_patterns = [
        r'```json\s*(\{[^`]+\})\s*```',
        r'```json\s*(\[[^\]]+\])\s*```',
        r'```\s*(\{[^`]+\})\s*```',
        r'```\s*(\[[^\]]+\])\s*```',
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            if debug:
                print(f"      [JSON Parse] Found in code block")
            try:
                return json.loads(json_str)
            except Exception as e:
                if debug:
                    print(f"      [JSON Parse] Code block parse failed: {e}")
    
    # Strategy 2: Balanced braces/brackets
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        brace_count = 0
        start_idx = None
        
        for i, char in enumerate(text):
            if char == start_char:
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == end_char:
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    json_str = text[start_idx:i+1]
                    if debug:
                        print(f"      [JSON Parse] Found balanced {start_char}{end_char}")
                    try:
                        return json.loads(json_str)
                    except Exception as e:
                        if debug:
                            print(f"      [JSON Parse] Balanced parse failed: {e}")
                    start_idx = None
    
    # Strategy 3: Parse entire text
    try:
        return json.loads(text)
    except:
        pass
    
    if debug:
        print(f"      [JSON Parse] All strategies failed")
        print(f"      [JSON Parse] Raw text: {text[:500]}...")
    
    return None


# ============================================================================
# RULE-BASED FILTERING UTILITIES
# ============================================================================

def is_only_numbers(text: str) -> bool:
    """Check if text contains only numbers (including spaces, commas, dots)"""
    if not text:
        return False
    # Remove common number separators
    cleaned = text.replace(' ', '').replace(',', '').replace('.', '').replace('-', '')
    return cleaned.isdigit() and len(cleaned) > 0


def is_only_special_chars(text: str) -> bool:
    """Check if text contains only special characters"""
    if not text:
        return False
    # Check if all characters are non-alphanumeric and non-Korean
    return all(not c.isalnum() and not ('\uAC00' <= c <= '\uD7A3') for c in text)


def is_only_alphabet(text: str) -> bool:
    """Check if text contains only English alphabet letters (ignoring spaces)"""
    if not text:
        return False
    cleaned = text.replace(' ', '')
    return cleaned.isalpha() and all(ord(c) < 128 for c in cleaned) and len(cleaned) > 0


def is_single_korean_char(text: str) -> bool:
    """Check if text is a single Korean character"""
    if not text:
        return False
    cleaned = text.strip()
    return len(cleaned) == 1 and '\uAC00' <= cleaned <= '\uD7A3'


def should_filter_by_rules(text: str) -> Tuple[bool, str]:
    """
    Check if text should be filtered by rules
    
    Returns:
        (should_filter, reason)
    """
    if not text or not text.strip():
        return True, "empty"
    
    text = text.strip()
    
    if is_only_numbers(text):
        return True, "only_numbers"
    
    if is_only_special_chars(text):
        return True, "only_special_chars"
    
    if is_only_alphabet(text):
        return True, "only_alphabet"
    
    if is_single_korean_char(text):
        return True, "single_korean_char"
    
    # Filter Korean POIs ending with generic terms
    if text.endswith("강"):  # River
        return True, "ends_with_강_river"
    
    if text.endswith("도로"):  # Road
        return True, "ends_with_도로_road"
    
    return False, ""


# ============================================================================
# ADVANCED FEATURE EXTRACTION & MATCHING UTILITIES
# ============================================================================

def extract_road_mask(image: np.ndarray) -> np.ndarray:
    """
    Extract road features from map image
    Roads typically appear as continuous lines with specific color ranges
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Binary mask highlighting roads
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Roads are often yellow/orange/white on maps
    # Define color ranges for roads
    lower_yellow = np.array([15, 30, 100])
    upper_yellow = np.array([35, 255, 255])
    
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])
    
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    
    # Create masks
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine masks
    road_mask = cv2.bitwise_or(mask_yellow, mask_white)
    road_mask = cv2.bitwise_or(road_mask, mask_orange)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    
    # Normalize to 0-1
    return road_mask.astype(np.float32) / 255.0


def extract_water_mask(image: np.ndarray) -> np.ndarray:
    """
    Extract water features from map image
    Water typically appears as blue regions
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Binary mask highlighting water bodies
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Water is typically blue on maps
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    lower_light_blue = np.array([85, 20, 100])
    upper_light_blue = np.array([135, 255, 255])
    
    # Create masks
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
    
    # Combine
    water_mask = cv2.bitwise_or(mask_blue, mask_light_blue)
    
    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    # Normalize
    return water_mask.astype(np.float32) / 255.0


def extract_edge_features(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract edge features using multiple methods
    
    Args:
        image: Input image (BGR)
        
    Returns:
        (canny_edges, sobel_magnitude, gradient_orientation)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Canny edges
    canny = cv2.Canny(gray, 50, 150)
    canny_norm = canny.astype(np.float32) / 255.0
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude_norm = magnitude / (magnitude.max() + 1e-6)
    
    # Orientation (in radians, -pi to pi)
    orientation = np.arctan2(sobely, sobelx)
    
    return canny_norm, magnitude_norm, orientation


def extract_multi_scale_features(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive multi-scale features from map image
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Dictionary of feature maps
    """
    print(f"      Extracting grayscale...")
    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray_norm = gray.astype(np.float32) / 255.0
    
    print(f"      Extracting road mask...")
    # Road mask
    road_mask = extract_road_mask(image)
    
    print(f"      Extracting water mask...")
    # Water mask
    water_mask = extract_water_mask(image)
    
    print(f"      Extracting edges (Canny, Sobel, orientation)...")
    # Edge features
    canny, sobel_mag, orientation = extract_edge_features(image)
    
    features = {
        'grayscale': gray_norm,
        'road_mask': road_mask,
        'water_mask': water_mask,
        'canny_edges': canny,
        'sobel_magnitude': sobel_mag,
        'orientation': orientation
    }
    
    return features


def compute_zncc(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Compute Zero-mean Normalized Cross-Correlation (ZNCC)
    
    Args:
        feat1, feat2: Feature maps (same size)
        
    Returns:
        ZNCC score (range: -1 to 1, higher is better)
    """
    if feat1.shape != feat2.shape:
        raise ValueError("Feature maps must have same shape")
    
    # Zero-mean
    feat1_zm = feat1 - np.mean(feat1)
    feat2_zm = feat2 - np.mean(feat2)
    
    # Compute correlation
    numerator = np.sum(feat1_zm * feat2_zm)
    denominator = np.sqrt(np.sum(feat1_zm**2) * np.sum(feat2_zm**2))
    
    if denominator < 1e-10:
        return 0.0
    
    zncc = numerator / denominator
    return float(zncc)


def compute_chamfer_distance(edges1: np.ndarray, edges2: np.ndarray) -> float:
    """
    Compute Chamfer distance between two edge maps
    Lower distance means better match
    
    Args:
        edges1, edges2: Binary edge maps (0-1 range)
        
    Returns:
        Chamfer distance (lower is better, converted to higher is better)
    """
    # Threshold to binary
    binary1 = (edges1 > 0.5).astype(np.uint8)
    binary2 = (edges2 > 0.5).astype(np.uint8)
    
    # Compute distance transforms
    dist1 = distance_transform_edt(1 - binary1)
    dist2 = distance_transform_edt(1 - binary2)
    
    # Chamfer distance (symmetric)
    # Distance from edges1 to edges2
    chamfer_1_to_2 = np.sum(dist2[binary1 > 0]) / (np.sum(binary1) + 1e-6)
    
    # Distance from edges2 to edges1
    chamfer_2_to_1 = np.sum(dist1[binary2 > 0]) / (np.sum(binary2) + 1e-6)
    
    # Average
    chamfer = (chamfer_1_to_2 + chamfer_2_to_1) / 2.0
    
    # Normalize by image diagonal
    diag = np.sqrt(edges1.shape[0]**2 + edges1.shape[1]**2)
    chamfer_norm = chamfer / diag
    
    # Return negative so higher is better (consistent with ZNCC)
    return -chamfer_norm


def compute_orientation_similarity(orient1: np.ndarray, orient2: np.ndarray,
                                  magnitude1: np.ndarray, magnitude2: np.ndarray) -> float:
    """
    Compute orientation similarity between two gradient orientation maps
    
    Args:
        orient1, orient2: Orientation maps (radians, -pi to pi)
        magnitude1, magnitude2: Gradient magnitudes (for weighting)
        
    Returns:
        Orientation similarity score (0 to 1, higher is better)
    """
    # Only consider pixels with significant gradients
    threshold = 0.1
    mask1 = magnitude1 > threshold
    mask2 = magnitude2 > threshold
    mask = mask1 & mask2
    
    if np.sum(mask) < 100:  # Too few valid pixels
        return 0.0
    
    # Compute orientation difference
    orient_diff = orient1 - orient2
    
    # Wrap to [-pi, pi]
    orient_diff = np.arctan2(np.sin(orient_diff), np.cos(orient_diff))
    
    # Absolute difference (0 to pi)
    orient_diff_abs = np.abs(orient_diff)
    
    # Convert to similarity (1 when same, 0 when opposite)
    similarity = 1.0 - (orient_diff_abs / np.pi)
    
    # Weight by gradient magnitudes
    weights = magnitude1 * magnitude2
    weights = weights / (np.sum(weights) + 1e-6)
    
    # Weighted average
    weighted_similarity = np.sum(similarity[mask] * weights[mask]) / (np.sum(weights[mask]) + 1e-6)
    
    return float(weighted_similarity)


def compute_composite_similarity(features1: Dict[str, np.ndarray],
                                features2: Dict[str, np.ndarray],
                                weights: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
    """
    Compute composite similarity using multiple metrics
    
    Args:
        features1, features2: Feature dictionaries
        weights: Optional weights for each metric
        
    Returns:
        (composite_score, individual_scores)
    """
    if weights is None:
        weights = {
            'zncc_grayscale': 0.15,
            'zncc_road': 0.20,
            'zncc_water': 0.10,
            'zncc_sobel': 0.15,
            'chamfer_canny': 0.25,
            'orientation': 0.15
        }
    
    scores = {}
    
    # ZNCC on various features
    scores['zncc_grayscale'] = compute_zncc(
        features1['grayscale'], 
        features2['grayscale']
    )
    
    scores['zncc_road'] = compute_zncc(
        features1['road_mask'],
        features2['road_mask']
    )
    
    scores['zncc_water'] = compute_zncc(
        features1['water_mask'],
        features2['water_mask']
    )
    
    scores['zncc_sobel'] = compute_zncc(
        features1['sobel_magnitude'],
        features2['sobel_magnitude']
    )
    
    # Chamfer distance on edges
    scores['chamfer_canny'] = compute_chamfer_distance(
        features1['canny_edges'],
        features2['canny_edges']
    )
    
    # Orientation similarity
    scores['orientation'] = compute_orientation_similarity(
        features1['orientation'],
        features2['orientation'],
        features1['sobel_magnitude'],
        features2['sobel_magnitude']
    )
    
    # Compute weighted composite score
    composite = 0.0
    total_weight = 0.0
    
    for metric, score in scores.items():
        weight = weights.get(metric, 0.0)
        composite += score * weight
        total_weight += weight
    
    if total_weight > 0:
        composite /= total_weight
    
    return composite, scores


def sliding_window_match_advanced(input_features: Dict[str, np.ndarray],
                                 vworld_features: Dict[str, np.ndarray],
                                 step_size: int = 5,
                                 weights: Dict[str, float] = None) -> Tuple[Tuple[int, int], float, np.ndarray, Dict]:
    """
    Find best matching position using advanced sliding window with multiple metrics
    
    Args:
        input_features: Features of input map
        vworld_features: Features of VWorld bbox (larger)
        step_size: Stride for sliding window (pixels)
        weights: Optional weights for composite scoring
        
    Returns:
        (best_position, best_score, score_map, best_individual_scores)
    """
    # Get dimensions from grayscale
    input_h, input_w = input_features['grayscale'].shape
    vworld_h, vworld_w = vworld_features['grayscale'].shape
    
    if input_h > vworld_h or input_w > vworld_w:
        raise ValueError("Input features larger than VWorld features")
    
    # Calculate search space
    y_range = vworld_h - input_h + 1
    x_range = vworld_w - input_w + 1
    
    # Initialize score map
    score_map = np.full((y_range, x_range), -np.inf)
    
    best_score = -np.inf
    best_pos = (0, 0)
    best_individual_scores = {}
    
    total_positions = ((y_range - 1) // step_size + 1) * ((x_range - 1) // step_size + 1)
    print(f"    Searching {total_positions} positions (step={step_size})...")
    print(f"    Using metrics: ZNCC (gray, road, water, sobel), Chamfer (edges), Orientation")
    
    tested = 0
    # Sliding window search
    for y in range(0, y_range, step_size):
        for x in range(0, x_range, step_size):
            # Extract windows from VWorld
            window_features = {}
            for key, feat_map in vworld_features.items():
                window_features[key] = feat_map[y:y+input_h, x:x+input_w]
            
            # Compute composite similarity
            score, individual_scores = compute_composite_similarity(
                input_features,
                window_features,
                weights
            )
            
            score_map[y, x] = score
            
            tested += 1
            if tested % 50 == 0:
                print(f"      Progress: {tested}/{total_positions} ({100*tested/total_positions:.1f}%) - Best: {best_score:.4f}")
            
            # Update best
            if score > best_score:
                best_score = score
                best_pos = (y, x)
                best_individual_scores = individual_scores
    
    print(f"    ✓ Search complete. Best score: {best_score:.4f} at position {best_pos}")
    print(f"    Individual scores at best position:")
    for metric, score in best_individual_scores.items():
        print(f"      {metric}: {score:.4f}")
    
    return best_pos, best_score, score_map, best_individual_scores


# ============================================================================
# HIGHLIGHT EXTRACTION UTILITIES
# ============================================================================


def extract_highlight_mask(image_path: str, output_dir: Path, 
                          checkpoint_path: str, threshold: float = 0.1,
                          image_size: int = 512) -> Dict:
    """
    Extract highlight mask using trained model
    
    Args:
        image_path: Input image path
        output_dir: Directory to save outputs
        checkpoint_path: Path to trained model checkpoint
        threshold: Sigmoid probability threshold
        image_size: Model input size (should match training)
        
    Returns:
        Dictionary with paths to outputs and mask data
    """
    print(f"\n  [Highlight Extraction] Loading model from: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model (matching the original working code)
    def build_model(device):
        from torchvision.models.segmentation import deeplabv3_resnet50
        model = deeplabv3_resnet50(weights=None, aux_loss=True)
        model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
        model.aux_classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
        return model.to(device)
    
    # Load model state
    state = torch.load(checkpoint_path, map_location=device)
    model = build_model(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    
    print(f"  [Highlight Extraction] Model loaded successfully")
    
    # Load and preprocess image (matching original code)
    from torchvision import transforms as T
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    print(f"  [Highlight Extraction] Processing image: {W}x{H}")
    
    # Resize for model (matching original code)
    rgb_small = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = normalize(to_tensor(rgb_small)).unsqueeze(0).to(device)
    
    # Inference (matching original code)
    with torch.no_grad():
        logits = model(x)["out"]  # [1,1,h,w]
        prob_small = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    # Threshold and resize to original size (matching original code)
    mask_small = (prob_small > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Keep largest component (matching original code)
    mask = keep_largest_component(mask, close_ks=3)
    
    # Save raw mask
    name = Path(image_path).stem
    mask_path = output_dir / f"{name}_highlight_mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"  [Highlight Extraction] Saved raw mask: {mask_path}")
    
    # Create overlay (matching original code)
    overlay = bgr.copy()
    overlay[mask == 255] = (0, 0, 255)
    blend = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)
    overlay_path = output_dir / f"{name}_highlight_overlay.png"
    cv2.imwrite(str(overlay_path), blend)
    print(f"  [Highlight Extraction] Saved overlay: {overlay_path}")
    
    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "mask": mask,
        "original_size": (W, H)
    }



def keep_largest_component(mask_bin_uint8, close_ks=3):
    """Keep only the largest connected component (main road)"""
    m = (mask_bin_uint8 > 0).astype(np.uint8) * 255
    if close_ks and close_ks > 1:
        k = np.ones((close_ks, close_ks), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(m)
    out[labels == largest] = 255
    return out


def skeletonize_highlight(mask: np.ndarray, output_dir: Path, name: str) -> Dict:
    """
    Skeletonize thick highlight mask to polyline
    
    Args:
        mask: Binary mask (0/255)
        output_dir: Directory to save outputs
        name: Base name for outputs
        
    Returns:
        Dictionary with skeleton data and paths
    """
    print(f"\n  [Skeletonization] Computing medial axis...")
    
    # Convert to binary
    binary = (mask > 0).astype(bool)
    
    # Compute medial axis (skeleton)
    skeleton, distance = medial_axis(binary, return_distance=True)
    
    # Convert to uint8
    skeleton_img = (skeleton * 255).astype(np.uint8)
    
    # Save skeleton
    skeleton_path = output_dir / f"{name}_skeleton.png"
    cv2.imwrite(str(skeleton_path), skeleton_img)
    print(f"  [Skeletonization] Saved skeleton: {skeleton_path}")
    
    # Extract skeleton pixels
    skeleton_pixels = np.column_stack(np.where(skeleton))  # [y, x] format
    skeleton_pixels = skeleton_pixels[:, [1, 0]]  # Convert to [x, y]
    
    print(f"  [Skeletonization] Extracted {len(skeleton_pixels)} skeleton pixels")
    
    # Create visualization with skeleton overlay
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis[skeleton] = (0, 255, 0)  # Green skeleton
    
    skeleton_vis_path = output_dir / f"{name}_skeleton_overlay.png"
    cv2.imwrite(str(skeleton_vis_path), vis)
    print(f"  [Skeletonization] Saved skeleton overlay: {skeleton_vis_path}")
    
    return {
        "skeleton_path": str(skeleton_path),
        "skeleton_overlay_path": str(skeleton_vis_path),
        "skeleton": skeleton,
        "skeleton_pixels": skeleton_pixels,
        "distance_transform": distance
    }


def smooth_polyline(points: np.ndarray, smoothing: float = 2.0, num_points: int = None) -> np.ndarray:
    """
    Smooth polyline using B-spline interpolation
    
    Args:
        points: Array of points [N, 2] (x, y)
        smoothing: Smoothing parameter (higher = smoother)
        num_points: Number of output points (if None, use input length)
        
    Returns:
        Smoothed polyline [M, 2]
    """
    if len(points) < 4:
        return points
    
    if num_points is None:
        num_points = len(points)
    
    try:
        # Fit B-spline
        tck, u = splprep([points[:, 0], points[:, 1]], s=smoothing, k=min(3, len(points)-1))
        
        # Evaluate at uniform intervals
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        
        smoothed = np.column_stack([x_new, y_new])
        return smoothed
    except:
        # Fallback: return original
        return points


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class RoadDetectionPipeline:
    """Complete end-to-end pipeline for road detection and GPS extraction"""
    
    def __init__(self, 
                 output_dir: str = "./road_detection_result",
                 database_path: str = None,
                 vworld_api_key: str = None,
                 llm_type: str = "qwen",
                 gpt_api_key: str = None,
                 reasoning_effort: str = "minimal",
                 matching_method: str = "affine"):
        """
        Initialize the pipeline
        
        Args:
            output_dir: Root directory for outputs
            database_path: Path to Korean Place Name Database (parquet)
            vworld_api_key: VWorld API key
            llm_type: "qwen" or "gpt"
            gpt_api_key: OpenAI API key (for GPT mode)
            reasoning_effort: GPT reasoning capacity ("minimal", "low", "medium", "high")
            matching_method: "affine" or "feature"
        """
        self.output_root = Path(output_dir)  # Changed from output_dir to output_root
        self.output_dir = None  # Will be set after image name is known
        self.reasoning_effort = reasoning_effort
        self.matching_method = matching_method
        
        # VWorld API key
        if vworld_api_key is None:
            vworld_key_file = PROJECT_DIR / "vworld_api_key"
            if vworld_key_file.exists():
                with open(vworld_key_file, 'r') as f:
                    content = f.read().strip()
                    if '=' in content:
                        self.vworld_api_key = content.split('=')[1].strip().strip("'\"")
                    else:
                        self.vworld_api_key = content
            else:
                self.vworld_api_key = '6F26555B-1DA0-3711-B567-E18B2260DB6D'
        else:
            self.vworld_api_key = vworld_api_key
        
        # Database
        if database_path is None:
            raise ValueError("database_path is required")
        
        self.database_path = Path(database_path)
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {database_path}")
        
        print(f"Loading database from: {self.database_path}")
        self.database = pd.read_parquet(self.database_path)
        
        # Create normalized column for fast searching
        print(f"  Creating normalized search index...")
        self.database['name_normalized'] = self.database['name'].apply(normalize_text_for_search)
        
        print(f"  ✓ Loaded {len(self.database):,} place names")
        
        # Load place.csv second column (for administrative containment filtering before bbox)
        try:
            place_csv_path = PROJECT_DIR / 'place.csv'
            if place_csv_path.exists():
                try:
                    place_df_tmp = pd.read_csv(place_csv_path, header=None, dtype=str, encoding='utf-8-sig')
                except Exception:
                    # fallback to common Korean encodings
                    try:
                        place_df_tmp = pd.read_csv(place_csv_path, header=None, dtype=str, encoding='euc-kr')
                    except Exception:
                        place_df_tmp = pd.read_csv(place_csv_path, header=None, dtype=str, encoding='cp949')

                # second column (index 1) usually contains the place name
                if place_df_tmp.shape[1] > 1:
                    second_col = place_df_tmp.iloc[:, 1].astype(str).fillna('')
                else:
                    # if only one column, use that as fallback
                    second_col = place_df_tmp.iloc[:, 0].astype(str).fillna('')

                # normalize for fast contains checks
                self.place_second_col_norm = [normalize_text_for_search(x) for x in second_col.tolist() if x and str(x).strip()]
                print(f"  ✓ Loaded place.csv ({len(self.place_second_col_norm)} names for administrative containment checks)")
            else:
                self.place_second_col_norm = []
                print(f"  ⚠ place.csv not found at {place_csv_path}; administrative containment checks disabled")
        except Exception as e:
            print(f"  ⚠ Error loading place.csv: {e}")
            self.place_second_col_norm = []

        # Initialize LLM
        print(f"Initializing LLM ({llm_type})...")
        llm_kwargs = {"reasoning_effort": reasoning_effort}
        if llm_type == "gpt":
            llm_kwargs['api_key'] = gpt_api_key
        self.llm = LLMWrapper(model_type=llm_type, **llm_kwargs)
        print("  ✓ LLM initialized")
        
        # Initialize OCR
        print("Initializing PaddleOCR v5...")
        self.ocr = PaddleOCR(
            lang='korean',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        print("  ✓ PaddleOCR initialized")
        
        # NOTE: setup_directories() will be called after image name is known
        
        # Pipeline state
        self.current_image_path = None
        self.image_name = None
        self.step_counter = 0
        self.dirs = {}  # Initialize empty, will be populated later
        
        # Track attempted combinations
        self.attempted_combinations: Set[frozenset] = set()

    def setup_directories(self):
        """Create directory structure under image-specific subdirectory"""
        # Create image-specific output directory
        self.output_dir = self.output_root / self.image_name
        
        self.dirs = {
            'root': self.output_dir,
            
            # Stage 1: OCR & POI Extraction
            'stage1_phase1': self.output_dir / '01_stage1_ocr_extraction',
            'stage1_phase2': self.output_dir / '02_stage1_llm_revision_all',
            'stage1_phase3': self.output_dir / '03_stage1_llm_revision_individual',
            'stage1_phase4': self.output_dir / '04_stage1_bbox_extraction',
            
            # Stage 2: Map Alignment
            'stage2': self.output_dir / '05_stage2_map_alignment',
            'stage2_features': self.output_dir / '05_stage2_map_alignment' / 'features',
            
            # Stage 3: Highlight Extraction
            'stage3': self.output_dir / '06_stage3_highlight_extraction',
            
            # Stage 4: Highlight Mapping
            'stage4': self.output_dir / '07_stage4_highlight_mapping',
            
            # Stage 5: GPS Calculation
            'stage5': self.output_dir / '08_stage5_gps_calculation',
            
            # Supporting directories
            'crops': self.output_dir / '01_stage1_ocr_extraction' / 'crops',
            'visualizations': self.output_dir / '01_stage1_ocr_extraction' / 'visualizations',
            'vworld_images': self.output_dir / '04_stage1_bbox_extraction' / 'vworld_images',  # FIXED: Now under stage1_phase4
            'logs': self.output_dir / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created output directory structure: {self.output_dir}")

    def save_artifact(self, data: dict, filename: str, phase: str = 'root'):
        """Save artifact as JSON"""
        filepath = self.dirs[phase] / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved: {filepath}")
        return filepath
    
    def log_step(self, message: str):
        """Log pipeline step"""
        self.step_counter += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] Step {self.step_counter}: {message}"
        print(f"\n{'='*80}")
        print(log_msg)
        print('='*80)
        
        log_file = self.dirs['logs'] / f"{self.image_name}_pipeline.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    # ========================================================================
    # PHASE 1: Initial OCR + Crop OCR + Rule-based Filter + LLM Filter + Ranking + DB Search
    # ========================================================================

    def phase1_initial_ocr(self, image_path: str) -> Dict:
        """Phase 1.1: Run initial OCR and save visualization"""
        
        self.current_image_path = Path(image_path)
        self.image_name = self.current_image_path.stem
        
        # Create directory structure now that we know the image name
        self.setup_directories()
        
        # Set up comprehensive logging - CAPTURE ALL OUTPUT
        log_file_path = self.dirs['logs'] / f"{self.image_name}_complete_output.log"
        self.log_file_handle = open(log_file_path, 'w', encoding='utf-8')
        self.tee_logger = TeeLogger(self.log_file_handle)
        sys.stdout = self.tee_logger
        
        print(f"\n{'='*80}")
        print(f"LOGGING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output will be saved to: {log_file_path}")
        print(f"{'='*80}\n")
        
        self.log_step("Phase 1.1 - Initial OCR Extraction")
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_h, img_w = img.shape[:2]
        
        print(f"Running PaddleOCR on: {image_path}")
        result = self.ocr.predict(str(image_path))
        
        if not result:
            empty = {
                "image_info": {"path": str(image_path), "width": img_w, "height": img_h},
                "extracted_texts": []
            }
            self.save_artifact(empty, f"{self.image_name}_1a_initial_ocr.json", 'stage1_phase1')
            return empty
        
        res_dict = result[0].json
        ocr_data = self.convert_ocr_to_structured(res_dict, str(image_path), img_w, img_h)
        
        print(f"  ✓ Initial OCR extracted {len(ocr_data['extracted_texts'])} texts")
        
        # DEBUG: Show first few extractions
        print(f"\n  [DEBUG] First 5 extractions:")
        for i, entry in enumerate(ocr_data['extracted_texts'][:5]):
            print(f"    {i+1}. Index {entry['index']}: '{entry['name']}' (conf: {entry['confidence']}%)")
        
        # Save visualization
        viz_path = self.dirs['visualizations'] / f"{self.image_name}_1a_initial_ocr.jpg"
        save_ocr_visualization(self.current_image_path, ocr_data, viz_path)
        
        self.save_artifact(ocr_data, f"{self.image_name}_1a_initial_ocr.json", 'stage1_phase1')
        
        return ocr_data


    def phase1_crop_ocr(self, ocr_data: Dict) -> Dict:
        """Phase 1.2: Crop-level OCR for all indices"""
        self.log_step("Phase 1.2 - Crop-level OCR for All Indices")
        
        if not ocr_data["extracted_texts"]:
            print("  No texts to crop")
            return ocr_data
        
        print(f"  Running crop-level OCR for {len(ocr_data['extracted_texts'])} indices...")
        
        for entry in ocr_data["extracted_texts"]:
            idx = entry["index"]
            original_name = entry["name"]
            original_conf = entry["confidence"]
            
            print(f"\n  Index {idx}: '{original_name}' (conf: {original_conf:.1f}%)")
            
            # Crop and save
            crop_path = self.crop_bounding_box(
                self.current_image_path,
                entry["bounding_box"],
                idx
            )
            
            try:
                # Run OCR on crop
                result = self.ocr.predict(str(crop_path))
                
                if result and result[0].json.get("res"):
                    res = result[0].json["res"]
                    rec_texts = res.get("rec_texts", [])
                    rec_scores = res.get("rec_scores", [])
                    
                    if rec_texts and rec_scores:
                        crop_name = rec_texts[0]
                        crop_conf = float(rec_scores[0]) * 100
                        
                        print(f"    Crop OCR: '{crop_name}' (conf: {crop_conf:.1f}%)")
                        
                        # Compare confidence
                        if crop_conf > original_conf:
                            print(f"    ✓ Using crop result (higher confidence)")
                            entry["name"] = crop_name
                            entry["confidence"] = round(crop_conf, 2)
                            entry["ocr_source"] = "crop"
                        else:
                            print(f"    → Keeping original (higher confidence)")
                            entry["ocr_source"] = "initial"
                    else:
                        print(f"    → Crop OCR empty, keeping original")
                        entry["ocr_source"] = "initial"
                else:
                    print(f"    → Crop OCR failed, keeping original")
                    entry["ocr_source"] = "initial"
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
                entry["ocr_source"] = "initial"
        
        self.save_artifact(ocr_data, f"{self.image_name}_1b_crop_ocr.json", 'stage1_phase1')
        
        return ocr_data
    
    def phase1_rule_based_filtering(self, ocr_data: Dict) -> Dict:
        """Phase 1.3: Rule-based filtering of non-useful texts"""
        self.log_step("Phase 1.3 - Rule-based Filtering")
        
        if not ocr_data["extracted_texts"]:
            print("  No texts to filter")
            ocr_data["rule_filtered_indices"] = []
            return ocr_data
        
        rule_filtered = []
        filter_reasons = {}
        
        print(f"  Applying rule-based filtering to {len(ocr_data['extracted_texts'])} texts...")
        
        # DEBUG: Show what we're filtering
        print(f"\n  [DEBUG] Checking each entry:")
        
        for entry in ocr_data["extracted_texts"]:
            idx = entry["index"]
            text = entry["name"]
            
            # Normalize for display
            text_normalized = normalize_text_simple(text)
            
            print(f"    Index {idx}: '{text_normalized}' (len: {len(text_normalized)})")
            
            should_filter, reason = should_filter_by_rules(text_normalized)
            
            if should_filter:
                rule_filtered.append(idx)
                filter_reasons[idx] = reason
                print(f"      ✗ FILTERED ({reason})")
            else:
                print(f"      ✓ Keep")
        
        ocr_data["rule_filtered_indices"] = rule_filtered
        ocr_data["rule_filter_reasons"] = filter_reasons
        
        original_count = len(ocr_data["extracted_texts"])
        
        # Create new list with only non-filtered entries
        filtered_texts = []
        for entry in ocr_data["extracted_texts"]:
            if entry["index"] not in rule_filtered:
                filtered_texts.append(entry)
        
        ocr_data["extracted_texts"] = filtered_texts
        
        print(f"\n  ✓ Rule-based filtering: {original_count} → {len(ocr_data['extracted_texts'])} (removed {len(rule_filtered)})")
        
        # DEBUG: Show remaining entries
        print(f"\n  [DEBUG] Remaining {len(ocr_data['extracted_texts'])} entries:")
        for i, entry in enumerate(ocr_data['extracted_texts'][:5]):
            print(f"    {i+1}. Index {entry['index']}: '{entry['name']}'")
        if len(ocr_data['extracted_texts']) > 5:
            print(f"    ... and {len(ocr_data['extracted_texts']) - 5} more")
        
        self.save_artifact(ocr_data, f"{self.image_name}_1c_rule_filtered.json", 'stage1_phase1')
        
        return ocr_data
    
    def phase1_llm_filtering(self, ocr_data: Dict) -> Dict:
        """Phase 1.4: LLM filtering of non-useful texts"""
        self.log_step("Phase 1.4 - LLM Filtering")
        
        if not ocr_data["extracted_texts"]:
            print("  No texts to filter")
            ocr_data["llm_filtered_indices"] = []
            return ocr_data
        
        # Build summary with validation
        texts_summary = []
        for entry in ocr_data["extracted_texts"]:
            text = normalize_text_simple(entry["name"])
            
            # Skip if actually empty after normalization
            if not text:
                print(f"  [WARNING] Index {entry['index']} has empty text after normalization!")
                continue
            
            texts_summary.append({
                "index": entry["index"],
                "text": text,
                "confidence": entry["confidence"]
            })
        
        if not texts_summary:
            print("  No valid texts for LLM filtering")
            ocr_data["llm_filtered_indices"] = []
            return ocr_data
        
        # DEBUG: Show what we're sending to LLM
        print(f"\n  [DEBUG] Sending {len(texts_summary)} texts to LLM:")
        for i, item in enumerate(texts_summary[:5]):
            print(f"    {i+1}. Index {item['index']}: '{item['text']}' (conf: {item['confidence']}%)")
        if len(texts_summary) > 5:
            print(f"    ... and {len(texts_summary) - 5} more")
        
        prompt = f"""This is a Korean map image. I extracted these text labels using OCR:

{json.dumps(texts_summary, ensure_ascii=False, indent=2)}

Identify which indices to REMOVE (not useful for locating the map region):
- Generic nouns (e.g., "고속도로", "도로", "길")
- Descriptive information that are used to explain the map (e.g., "100m", "2km", "과업 구간")
- Directions (e.g., "북쪽", "남쪽", "동", "서")
- Common facility names (e.g., "주차장", "화장실", "공원")
- Large administrative names (e.g., entire provinces like "경기도", "부산", "서울특별시", or large districts like "강남구", "서초구")
- Single characters (e.g., "공", "길", "로")
- Pure numbers (e.g., "12", "23")
- Random characters/noise (e.g., "@", "#$%")

IMPORTANT: Only remove indices that are truly not useful. DO NOT remove valid Korean place names.

Respond with ONLY this JSON format (no other text):
{{
    "indices_to_remove": [list of index numbers]
}}"""
        
        print(f"  Filtering {len(texts_summary)} texts via LLM...")
        
        # Text-only inference (no image needed)
        response = self.llm.inference(None, prompt, max_tokens=512)
        
        if not response:
            print(f"  ✗ LLM filtering failed, keeping all texts")
            ocr_data["llm_filtered_indices"] = []
            return ocr_data
        
        filter_result = extract_json_from_text(response, debug=True)
        
        if not filter_result:
            print(f"  ✗ Could not parse LLM response")
            print(f"  Raw: {response[:300]}...")
            ocr_data["llm_filtered_indices"] = []
            return ocr_data
        
        indices_to_remove = filter_result.get("indices_to_remove", [])
        
        # Validate indices exist in current data
        valid_indices = {e["index"] for e in ocr_data["extracted_texts"]}
        indices_to_remove = [idx for idx in indices_to_remove if idx in valid_indices]
        
        print(f"  ✓ Removing {len(indices_to_remove)} indices:")
        for idx in indices_to_remove:
            text = next((e.get("name", "") for e in texts_summary if e.get("index") == idx), "")
            print(f"    - Index {idx}: '{text}'")
        
        ocr_data["llm_filtered_indices"] = indices_to_remove
        
        original_count = len(ocr_data["extracted_texts"])
        
        # Create new list with only non-filtered entries
        filtered_texts = []
        for entry in ocr_data["extracted_texts"]:
            if entry["index"] not in indices_to_remove:
                filtered_texts.append(entry)
        
        ocr_data["extracted_texts"] = filtered_texts
        
        print(f"  ✓ LLM filtering: {original_count} → {len(ocr_data['extracted_texts'])}")
        
        # DEBUG: Show remaining entries
        print(f"\n  [DEBUG] Remaining {len(ocr_data['extracted_texts'])} entries after LLM filter:")
        for i, entry in enumerate(ocr_data['extracted_texts'][:5]):
            print(f"    {i+1}. Index {entry['index']}: '{entry['name']}'")
        if len(ocr_data['extracted_texts']) > 5:
            print(f"    ... and {len(ocr_data['extracted_texts']) - 5} more")
        
        self.save_artifact(ocr_data, f"{self.image_name}_1d_llm_filtered.json", 'stage1_phase1')
        
        return ocr_data
    
    def phase1_llm_ranking(self, ocr_data: Dict) -> Dict:
        """Phase 1.5: LLM ranking by usefulness"""
        self.log_step("Phase 1.5 - LLM Ranking by Usefulness")
        
        if not ocr_data["extracted_texts"]:
            print("  No texts to rank")
            return ocr_data
        
        texts_for_ranking = []
        for entry in ocr_data["extracted_texts"]:
            texts_for_ranking.append({
                "index": entry["index"],
                "text": entry["name"]
            })
        
        prompt = f"""These are Korean place names extracted from a map:

{json.dumps(texts_for_ranking, ensure_ascii=False, indent=2)}

Rank them by usefulness for locating the map region. Consider:
- More specific place names rank higher (e.g., "청계천" better than "서울")
- Unique landmarks rank higher than generic ones
- Small/specific areas rank higher than large/broad areas
- IMPORTANT: Prioritize very specific places 
- IMPORTANT: Deprioritize large administrative units:
  * Places ending with 도 (province) rank VERY LOW
  * Places ending with 시 (city) rank LOW
  * Places ending with 군 (county) rank LOW
  * Places ending with 구 (district) rank LOW
  * Places ending with 면 (township) rank LOWER
  * Places ending with 동/리 (neighborhood/village) rank HIGHER
  * Specific landmarks (small villages, specific building names, mountain names, local parks, shops, IC, etc) rank HIGHEST

Return ONLY a JSON array of indices, ordered from most useful (rank 1) to least useful:
[3, 7, 1, 5, ...]

No other text, just the array."""
        
        print(f"  Ranking {len(texts_for_ranking)} texts via LLM...")
        
        response = self.llm.inference(None, prompt, max_tokens=512)
        
        if not response:
            print(f"  ✗ LLM ranking failed, using original order")
            for i, entry in enumerate(ocr_data["extracted_texts"]):
                entry["usefulness_rank"] = i
            return ocr_data
        
        ranked_array = extract_json_from_text(response, debug=True)
        
        if not ranked_array or not isinstance(ranked_array, list):
            print(f"  ✗ Could not parse ranking")
            print(f"  Raw: {response[:300]}...")
            for i, entry in enumerate(ocr_data["extracted_texts"]):
                entry["usefulness_rank"] = i
            return ocr_data
        
        print(f"  ✓ LLM ranked indices: {ranked_array[:10]}{'...' if len(ranked_array) > 10 else ''}")
        
        # Assign ranks
        for entry in ocr_data["extracted_texts"]:
            idx = entry["index"]
            if idx in ranked_array:
                entry["usefulness_rank"] = ranked_array.index(idx)
            else:
                entry["usefulness_rank"] = len(ranked_array)
        
        # Show top 5
        sorted_entries = sorted(ocr_data["extracted_texts"], key=lambda e: e["usefulness_rank"])
        print(f"\n  Top 5 most useful:")
        for i, entry in enumerate(sorted_entries[:5], 1):
            print(f"    {i}. Index {entry['index']}: '{entry['name']}'")
        
        self.save_artifact(ocr_data, f"{self.image_name}_1e_ranked.json", 'stage1_phase1')
        
        return ocr_data
    
    def phase1_database_search(self, ocr_data: Dict) -> Dict:
        """Phase 1.6: Database search (normalized exact match)"""
        self.log_step("Phase 1.6 - Database Search (Normalized)")
        
        extraction_data = {
            "image_info": ocr_data["image_info"],
            "rule_filtered_indices": ocr_data.get("rule_filtered_indices", []),
            "llm_filtered_indices": ocr_data.get("llm_filtered_indices", []),
            "extracted_texts": [],
            "summary": {}
        }
        
        print(f"\n  [DEBUG] Database search for {len(ocr_data['extracted_texts'])} entries:")
        
        for entry in ocr_data["extracted_texts"]:
            place_name = entry.get("name", "").strip()
            place_name_normalized = normalize_text_simple(place_name)
            
            extraction_entry = {
                "index": entry["index"],
                "place_name": place_name_normalized,
                "original_ocr_text": place_name_normalized,
                "confidence": entry["confidence"],
                "ocr_source": entry.get("ocr_source", "initial"),
                "usefulness_rank": entry.get("usefulness_rank", 999),
                "pixel_center": entry["pixel_center"],
                "bounding_box": entry["bounding_box"],
                "existence_status": 0,
                "gps_coords_epsg5179": [],
                "database_matches": [],
                "correction_history": []
            }
            
            if place_name_normalized:
                matches = self.search_database_normalized(place_name_normalized)
                
                if len(matches) > 0:
                    print(f"  ✓ Index {entry['index']}: '{place_name_normalized}' - {len(matches)} match(es) (rank: {extraction_entry['usefulness_rank']})")
                    extraction_entry["existence_status"] = 1
                    extraction_entry["database_matches"] = matches
                    extraction_entry["gps_coords_epsg5179"] = [
                        {"x": float(m['x']), "y": float(m['y']),
                         "name": m['name'], "address": m.get('address_old', '')}
                        for m in matches
                    ]
                else:
                    print(f"  ✗ Index {entry['index']}: '{place_name_normalized}' - Not found (rank: {extraction_entry['usefulness_rank']})")
                    # DEBUG: Show normalized query
                    query_norm = normalize_text_for_search(place_name_normalized)
                    print(f"     [DEBUG] Normalized query: '{query_norm}'")
            else:
                print(f"  ✗ Index {entry['index']}: EMPTY after normalization!")
            
            extraction_data["extracted_texts"].append(extraction_entry)
        
        extraction_data["summary"] = {
            "total_extractions": len(extraction_data["extracted_texts"]),
            "found_in_db": sum(1 for e in extraction_data["extracted_texts"] if e["existence_status"] == 1),
            "not_found": sum(1 for e in extraction_data["extracted_texts"] if e["existence_status"] == 0)
        }
        
        self.save_artifact(extraction_data, f"{self.image_name}_1f_db_search.json", 'stage1_phase1')
        
        print(f"\n✓ Phase 1: {extraction_data['summary']['found_in_db']}/{extraction_data['summary']['total_extractions']} found")
        
        return extraction_data
    
    def convert_ocr_to_structured(self, result_dict: dict, image_path: str,
                                   img_width: int, img_height: int) -> Dict:
        """Convert PaddleOCR output to structured format"""
        res = result_dict.get("res", result_dict)
        rec_texts = res.get("rec_texts", []) or []
        rec_scores = res.get("rec_scores", []) or []
        rec_polys = res.get("rec_polys", []) or []
        
        extracted_texts = []
        
        for idx, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys), start=1):
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()
            if not isinstance(poly, (list, tuple)) or len(poly) < 4:
                continue
            
            top_left = {"x": float(poly[0][0]), "y": float(poly[0][1])}
            top_right = {"x": float(poly[1][0]), "y": float(poly[1][1])}
            bottom_right = {"x": float(poly[2][0]), "y": float(poly[2][1])}
            bottom_left = {"x": float(poly[3][0]), "y": float(poly[3][1])}
            
            cx = (top_left["x"] + top_right["x"] + bottom_right["x"] + bottom_left["x"]) / 4.0
            cy = (top_left["y"] + top_right["y"] + bottom_right["y"] + bottom_left["y"]) / 4.0
            
            confidence = round(float(score) * 100, 2)
            
            # Normalize text
            text_normalized = normalize_text_simple(text)
            
            text_entry = {
                "index": idx,
                "name": text_normalized,
                "confidence": confidence,
                "pixel_center": {"x": round(cx, 1), "y": round(cy, 1)},
                "bounding_box": {
                    "top_left": top_left,
                    "top_right": top_right,
                    "bottom_right": bottom_right,
                    "bottom_left": bottom_left
                }
            }
            
            extracted_texts.append(text_entry)
        
        return {
            "image_info": {
                "path": str(image_path),
                "width": img_width,
                "height": img_height
            },
            "extracted_texts": extracted_texts
        }
    
    def search_database_normalized(self, query: str) -> List[Dict]:
        """
        Normalized search with whitespace removal and case-insensitivity
        """
        if not query or not query.strip():
            return []
        
        try:
            # Normalize query
            query_norm = normalize_text_for_search(query)
            
            if not query_norm:
                return []
            
            # Search in normalized column
            mask = self.database['name_normalized'] == query_norm
            matches = self.database[mask]
            
            result = []
            for _, row in matches.iterrows():
                result.append({
                    'id': row['id'],
                    'name': row['name'],
                    'category': row.get('catagory', ''),
                    'source': row.get('source', ''),
                    'address_old': row.get('address_old', ''),
                    'x': row['x'],
                    'y': row['y']
                })
            
            return result
            
        except Exception as e:
            print(f"  [ERROR] Database search failed: {e}")
            return []

    def filter_place_csv_second_column(self, extraction_data: Dict, phase_label: str) -> Dict:
        """
        Remove extraction entries whose normalized place_name is contained in any
        normalized second-column value from place.csv.

        This is intended to remove broad administrative names like '부산', '경기' if
        they appear inside detailed place names such as '부산광역시 중구'.
        Saves a debug artifact and updates extraction_data['summary'].
        """
        if not getattr(self, 'place_second_col_norm', None):
            # nothing to filter
            return extraction_data

        removed = []
        remaining = []

        for e in extraction_data.get('extracted_texts', []):
            pname = e.get('place_name') or e.get('name') or ''
            if not pname or not str(pname).strip():
                remaining.append(e)
                continue

            pnorm = normalize_text_for_search(str(pname))
            if not pnorm:
                remaining.append(e)
                continue

            # If normalized place name is contained in any normalized place.csv second-col entry => remove
            matched = False
            for csv_name in self.place_second_col_norm:
                if pnorm in csv_name:
                    matched = True
                    break

            if matched:
                removed.append({
                    'index': e.get('index'),
                    'place_name': pname,
                })
            else:
                remaining.append(e)

        extraction_data['extracted_texts'] = remaining

        # Update summary
        extraction_data['summary'] = {
            'total_extractions': len(extraction_data.get('extracted_texts', [])),
            'found_in_db': sum(1 for x in extraction_data.get('extracted_texts', []) if x.get('existence_status') == 1),
            'not_found': sum(1 for x in extraction_data.get('extracted_texts', []) if x.get('existence_status') == 0)
        }

        # Save artifact for debugging - determine correct directory
        phase_dir_map = {
            'phase1': 'stage1_phase1',
            'phase2': 'stage1_phase2',
            'phase3_final': 'stage1_phase3'
        }
        
        save_dir = phase_dir_map.get(phase_label, 'stage1_phase1')
        
        try:
            artifact = {
                'removed_by_place_csv_second_col': removed,
                'remaining_count': len(remaining),
                'phase': phase_label
            }
            self.save_artifact(artifact, f"{self.image_name}_placecsv_filtered_{phase_label}.json", save_dir)
            print(f"  ✓ place.csv containment filter after {phase_label}: removed {len(removed)} entries")
        except Exception:
            pass

        return extraction_data
    
    def crop_bounding_box(self, image_path: Path, bbox: Dict, index: int) -> Path:
        """Crop image using bounding box and save"""
        img = cv2.imread(str(image_path))
        
        tl, tr = bbox["top_left"], bbox["top_right"]
        br, bl = bbox["bottom_right"], bbox["bottom_left"]
        
        x_coords = [tl["x"], tr["x"], br["x"], bl["x"]]
        y_coords = [tl["y"], tr["y"], br["y"], bl["y"]]
        
        x_min = int(max(0, min(x_coords) - 5))
        x_max = int(min(img.shape[1], max(x_coords) + 5))
        y_min = int(max(0, min(y_coords) - 5))
        y_max = int(min(img.shape[0], max(y_coords) + 5))
        
        cropped = img[y_min:y_max, x_min:x_max]
        
        crop_path = self.dirs['crops'] / f"{self.image_name}_index{index}_crop.jpg"
        cv2.imwrite(str(crop_path), cropped)
        
        return crop_path
    
    # ========================================================================
    # PHASE 2: All-in-One LLM Revision
    # ========================================================================
    
    def phase2_all_in_one_revision(self, extraction_data: Dict) -> Dict:
        """Phase 2: All-in-one LLM revision for unmatched entries"""
        self.log_step("Phase 2 - All-in-One LLM Revision")
        
        unmatched = [e for e in extraction_data["extracted_texts"] if e["existence_status"] == 0]
        
        if not unmatched:
            print("  No unmatched entries")
            return extraction_data
        
        print(f"  Revising {len(unmatched)} unmatched entries via LLM...")
        
        # Prepare summary
        unmatched_summary = []
        for entry in unmatched:
            unmatched_summary.append({
                "index": entry["index"],
                "text": entry["place_name"]
            })
        
        prompt = f"""This is a Korean map. These place names were extracted by OCR but are not found in the Korean place name database, likely due to OCR errors:

{json.dumps(unmatched_summary, ensure_ascii=False, indent=2)}

Please correct these OCR errors to identify the actual Korean place names. Consider:
- Common OCR mistakes (similar-looking Korean characters)
- Context from the map image
- Typical Korean place name patterns
- Try other typical form of the Korean place names. For example, 고교 -> 고등학교 etc.

Respond with ONLY this JSON format (no other text):
{{
    "revisions": [
        {{"index": <index_number>, "corrected_name": "<corrected Korean name>"}},
        ...
    ]
}}

Only include indices that you can confidently correct."""
        
        response = self.llm.inference(str(self.current_image_path), prompt, max_tokens=1024)
        
        if not response:
            print(f"  ✗ LLM revision failed")
            return extraction_data
        
        revision_result = extract_json_from_text(response, debug=True)
        
        if not revision_result or "revisions" not in revision_result:
            print(f"  ✗ Could not parse LLM response")
            print(f"  Raw: {response[:300]}...")
            return extraction_data
        
        revisions = revision_result.get("revisions", [])
        
        print(f"  ✓ LLM suggested {len(revisions)} revisions")
        
        # Apply revisions
        revised_count = 0
        newly_found = 0
        
        for rev in revisions:
            idx = rev.get("index")
            corrected = normalize_text_simple(rev.get("corrected_name", "").strip())
            
            if not corrected:
                continue
            
            # Find entry
            entry = next((e for e in extraction_data["extracted_texts"] if e["index"] == idx), None)
            
            if not entry:
                continue
            
            original = entry["place_name"]
            
            print(f"\n  Index {idx}: '{original}' → '{corrected}'")
            
            # Search database with normalized search
            matches = self.search_database_normalized(corrected)
            
            success = len(matches) > 0
            
            entry['correction_history'].append({
                "method": "all_in_one_llm",
                "from": original,
                "to": corrected,
                "success": success
            })
            
            if success:
                print(f"    ✓ Found ({len(matches)} match(es))")
                entry["place_name"] = corrected
                entry["existence_status"] = 1
                entry["database_matches"] = matches
                entry["gps_coords_epsg5179"] = [
                    {"x": float(m['x']), "y": float(m['y']),
                     "name": m['name'], "address": m.get('address_old', '')}
                    for m in matches
                ]
                newly_found += 1
            else:
                print(f"    ✗ Not found")
                # DEBUG
                query_norm = normalize_text_for_search(corrected)
                print(f"     [DEBUG] Normalized: '{query_norm}'")
            
            revised_count += 1
        
        print(f"\n  ✓ Revised {revised_count} entries, found {newly_found} new matches")
        
        extraction_data["summary"] = {
            "total_extractions": len(extraction_data["extracted_texts"]),
            "found_in_db": sum(1 for e in extraction_data["extracted_texts"] if e["existence_status"] == 1),
            "not_found": sum(1 for e in extraction_data["extracted_texts"] if e["existence_status"] == 0)
        }
        
        self.save_artifact(extraction_data, f"{self.image_name}_2_all_in_one_revision.json", 'stage1_phase2')
        
        print(f"\n✓ Phase 2: {extraction_data['summary']['found_in_db']}/{extraction_data['summary']['total_extractions']} found")
        
        return extraction_data
    
    # ========================================================================
    # PHASE 3: Individual Crop-level LLM Revision
    # ========================================================================
    
    def phase3_individual_revision(self, extraction_data: Dict, max_attempts: int = 5) -> Dict:
        """Phase 3: Individual crop-level LLM revision"""
        self.log_step("Phase 3 - Individual Crop-level LLM Revision")
        
        unmatched = [e for e in extraction_data["extracted_texts"] if e["existence_status"] == 0]
        
        if not unmatched:
            print("  No unmatched entries")
            return extraction_data
        
        # Sort by usefulness rank (try most useful first)
        unmatched_sorted = sorted(unmatched, key=lambda e: e.get("usefulness_rank", 999))
        
        print(f"  Individually revising {len(unmatched_sorted)} entries (up to {max_attempts} attempts each)...")
        
        for entry in unmatched_sorted:
            idx = entry["index"]
            original = entry["original_ocr_text"]
            current = entry["place_name"]
            rank = entry.get("usefulness_rank", "N/A")
            
            print(f"\n  Index {idx}: '{current}' (rank: {rank})")
            
            crop_path = self.dirs['crops'] / f"{self.image_name}_index{idx}_crop.jpg"
            
            if not crop_path.exists():
                crop_path = self.crop_bounding_box(
                    self.current_image_path,
                    entry["bounding_box"],
                    idx
                )
            
            # Collect previous attempts
            previous_attempts = [original, current]
            for hist in entry.get('correction_history', []):
                if hist.get('to'):
                    previous_attempts.append(hist['to'])
            
            # Remove duplicates while preserving order
            seen = set()
            previous_attempts = [x for x in previous_attempts if not (x in seen or seen.add(x))]
            
            for attempt in range(max_attempts):
                print(f"    Attempt {attempt + 1}/{max_attempts}...")
                
                exclusion_str = ", ".join([f"'{p}'" for p in previous_attempts])
                
                prompt = f"""This is a cropped region from a Korean map showing a place name.

Original OCR: "{original}"
Failed attempts (DON'T suggest these): {exclusion_str}

Identify the correct Korean place name. Consider:
- Common OCR errors (similar-looking Korean characters)
- Context from this map region
- Typical Korean place name patterns

Respond with ONLY this JSON:
{{
    "place_name": "<corrected Korean name>",
    "confidence": <0-100>,
    "reasoning": "<brief explanation>"
}}"""
                
                response = self.llm.inference(str(crop_path), prompt, max_tokens=256)
                
                if not response:
                    print(f"      ✗ LLM failed")
                    continue
                
                correction = extract_json_from_text(response, debug=True)
                
                if not correction:
                    print(f"      ✗ Parse failed")
                    print(f"      Raw: {response[:200]}...")
                    continue
                
                corrected_name = normalize_text_simple(correction.get('place_name', '').strip())
                
                if not corrected_name or corrected_name in previous_attempts:
                    print(f"      ✗ Duplicate/empty: '{corrected_name}'")
                    continue
                
                previous_attempts.append(corrected_name)
                
                matches = self.search_database_normalized(corrected_name)
                
                success = len(matches) > 0
                
                entry['correction_history'].append({
                    "method": f"individual_llm_attempt_{attempt+1}",
                    "from": current,
                    "to": corrected_name,
                    "success": success
                })
                
                if success:
                    print(f"      ✓ Found: '{corrected_name}' ({len(matches)} match(es))")
                    entry["place_name"] = corrected_name
                    entry["existence_status"] = 1
                    entry["database_matches"] = matches
                    entry["gps_coords_epsg5179"] = [
                        {"x": float(m['x']), "y": float(m['y']),
                         "name": m['name'], "address": m.get('address_old', '')}
                        for m in matches
                    ]
                    
                    # Update summary
                    extraction_data["summary"]["found_in_db"] += 1
                    extraction_data["summary"]["not_found"] -= 1
                    
                    # Try bbox immediately after finding a new match
                    if extraction_data["summary"]["found_in_db"] >= 3:
                        print(f"\n      → Now have {extraction_data['summary']['found_in_db']} matches, trying bbox...")
                        bbox_result = self.try_bounding_box_extraction(extraction_data)
                        if bbox_result:
                            return extraction_data
                    
                    break
                else:
                    print(f"      ✗ Not found: '{corrected_name}'")
                    # DEBUG
                    query_norm = normalize_text_for_search(corrected_name)
                    print(f"       [DEBUG] Normalized: '{query_norm}'")
            
            if entry['existence_status'] == 0:
                print(f"    ✗ Failed all attempts")
        
        extraction_data["summary"] = {
            "total_extractions": len(extraction_data["extracted_texts"]),
            "found_in_db": sum(1 for e in extraction_data["extracted_texts"] if e["existence_status"] == 1),
            "not_found": sum(1 for e in extraction_data["extracted_texts"] if e["existence_status"] == 0)
        }
        
        self.save_artifact(extraction_data, f"{self.image_name}_3_individual_revision.json", 'stage1_phase3')
        
        print(f"\n✓ Phase 3: {extraction_data['summary']['found_in_db']}/{extraction_data['summary']['total_extractions']} found")
        
        return extraction_data
    
    # ========================================================================
    # BOUNDING BOX EXTRACTION
    # ========================================================================
    
    def try_bounding_box_extraction(self, extraction_data: Dict) -> Optional[Dict]:
        """Try to extract bounding box with smart combination testing"""
        confirmed = [e for e in extraction_data["extracted_texts"] if e["existence_status"] == 1]
        
        if len(confirmed) < 3:
            return None
        
        print(f"\n  → Trying bbox extraction with {len(confirmed)} confirmed POIs...")
        
        # Categorize by duplicates
        no_dup = [e for e in confirmed if len(e['gps_coords_epsg5179']) == 1]
        has_dup = [e for e in confirmed if len(e['gps_coords_epsg5179']) > 1]
        
        print(f"    - No duplicates: {len(no_dup)}")
        print(f"    - With duplicates: {len(has_dup)}")
        
        # Sort both by usefulness rank
        no_dup_sorted = sorted(no_dup, key=lambda e: e.get("usefulness_rank", 999))
        has_dup_sorted = sorted(has_dup, key=lambda e: e.get("usefulness_rank", 999))
        
        # Strategy: Prioritize no-dup + high-rank
        if len(no_dup_sorted) >= 3:
            print(f"\n  Testing no-duplicate combinations (rank-ordered)...")
            result = self.try_poi_combinations_ranked(no_dup_sorted, extraction_data, "no_duplicate_ranked")
            if result:
                return result
        
        # Mixed strategy
        if len(no_dup_sorted) >= 1 and len(has_dup_sorted) >= 1:
            print(f"\n  Testing mixed combinations (rank-ordered)...")
            mixed = no_dup_sorted + has_dup_sorted
            result = self.try_poi_combinations_ranked(mixed, extraction_data, "mixed_ranked")
            if result:
                return result
        
        # All POIs (resolve duplicates)
        print(f"\n  Testing all POIs with duplicate resolution (rank-ordered)...")
        all_sorted = sorted(confirmed, key=lambda e: e.get("usefulness_rank", 999))
        result = self.try_poi_combinations_ranked(all_sorted, extraction_data, "all_resolved_ranked")
        
        return result
    
    def try_poi_combinations_ranked(self, poi_list: List[Dict], extraction_data: Dict,
                                   strategy: str, max_attempts: int = 20) -> Optional[Dict]:
        """Try POI combinations with rank priority and duplicate tracking"""
        if len(poi_list) < 3:
            return None
        
        # Generate all 3-combinations
        all_combos = list(combinations(poi_list, 3))
        
        # Sort by average rank (lower is better)
        all_combos_sorted = sorted(all_combos, key=lambda combo: sum(p.get("usefulness_rank", 999) for p in combo))
        
        print(f"    Generated {len(all_combos_sorted)} combinations, testing up to {max_attempts}...")
        
        tested = 0
        for combo in all_combos_sorted:
            if tested >= max_attempts:
                break
            
            # Check if already attempted
            combo_key = frozenset([p['index'] for p in combo])
            
            # For duplicates, also track GPS coords
            combo_gps_key = frozenset([
                (p['index'], tuple(sorted((c['x'], c['y']) for c in p['gps_coords_epsg5179'])))
                for p in combo
            ])
            
            if combo_key in self.attempted_combinations or combo_gps_key in self.attempted_combinations:
                continue
            
            self.attempted_combinations.add(combo_key)
            self.attempted_combinations.add(combo_gps_key)
            
            tested += 1
            
            indices = [p['index'] for p in combo]
            names = [p['place_name'] for p in combo]
            ranks = [p.get('usefulness_rank', 'N/A') for p in combo]
            
            print(f"\n    Attempt {tested}: Indices {indices}")
            print(f"      Names: {names}")
            print(f"      Ranks: {ranks}")
            
            # Resolve duplicates
            pois_with_coords = self.resolve_duplicates_for_combo(
                list(combo),
                extraction_data['image_info']
            )
            
            if len(pois_with_coords) < 2:
                print(f"      ✗ Not enough resolved POIs")
                continue
            
            img_size = (extraction_data['image_info']['width'],
                       extraction_data['image_info']['height'])
            
            center_gps, meters_per_pixel = self.calculate_scale_and_center_gps(
                img_size, pois_with_coords
            )
            
            if not center_gps or not meters_per_pixel:
                print(f"      ✗ Could not calculate center/scale")
                continue
            
            print(f"      Center: ({center_gps[0]:.6f}, {center_gps[1]:.6f})")
            print(f"      Scale: {meters_per_pixel:.2f} m/px")
            
            if meters_per_pixel > 200.0:
                print(f"      ✗ Scale > 200 m/px ({meters_per_pixel:.2f}), skip this combo as invalid")
                continue
            
            zoom_level = self.calculate_zoom_level(center_gps[1], meters_per_pixel, img_size[0])
            
            # Adjust size
            pad_pct = 0.15
            orig_w, orig_h = img_size
            pad_w = max(int(orig_w * pad_pct), 20)
            pad_h = max(int(orig_h * pad_pct), 20)
            padded_size = (orig_w + pad_w * 2, orig_h + pad_h * 2)
            
            max_size = 1024
            cur_zoom = zoom_level
            cur_size = list(padded_size)
            
            while cur_size[0] > max_size or cur_size[1] > max_size:
                if cur_zoom <= 7:
                    scale = min(max_size / cur_size[0], max_size / cur_size[1])
                    cur_size[0] = int(cur_size[0] * scale)
                    cur_size[1] = int(cur_size[1] * scale)
                    break
                cur_zoom -= 1
                cur_size[0] = int(cur_size[0] / 2)
                cur_size[1] = int(cur_size[1] / 2)
            
            print(f"      Zoom: {cur_zoom}, Size: {tuple(cur_size)}")
            
            vworld_path = self.dirs['vworld_images'] / f"{self.image_name}_attempt{tested}_vworld.png"
            success = self.save_vworld_image(center_gps, cur_zoom, tuple(cur_size), vworld_path)
            
            if not success:
                print(f"      ✗ VWorld API failed")
                continue
            
            print(f"      Validating with LLM...")
            validation = self.validate_with_llm(
                str(self.current_image_path),
                str(vworld_path)
            )
            
            if validation and validation.get('matches', '').lower() == 'yes':
                print(f"      ✓ VALIDATED! (confidence: {validation.get('confidence', 0)}%)")
                
                # Calculate corner GPS
                corner_gps = self.calculate_bbox_corners(center_gps, cur_zoom, tuple(cur_size))
                
                # Convert POI GPS to VWorld pixel coordinates for visualization
                vworld_poi_pixels = []
                for poi in pois_with_coords:
                    vworld_x, vworld_y = self.gps_to_vworld_pixel(
                        poi['gps_lon'],
                        poi['gps_lat'],
                        center_gps[0],
                        center_gps[1],
                        cur_zoom,
                        tuple(cur_size)
                    )
                    vworld_poi_pixels.append({
                        'index': poi['index'],
                        'name': poi['name'],
                        'x': vworld_x,
                        'y': vworld_y
                    })
                
                # Save marked VWorld image
                marked_vworld_path = self.dirs['vworld_images'] / f"{self.image_name}_attempt{tested}_vworld_marked.png"
                save_marked_vworld_image(vworld_path, vworld_poi_pixels, marked_vworld_path)
                
                bbox_result = {
                    "image_info": extraction_data['image_info'],
                    "strategy": strategy,
                    "attempt": tested,
                    "confirmed_pois": pois_with_coords,
                    "vworld_poi_pixels": vworld_poi_pixels,  # POI positions in VWorld image
                    "center_gps": {"lon": center_gps[0], "lat": center_gps[1]},
                    "corner_gps": corner_gps,
                    "scale_meters_per_pixel": meters_per_pixel,
                    "zoom_level": cur_zoom,
                    "bbox_size": {"width": cur_size[0], "height": cur_size[1]},
                    "vworld_image_path": str(vworld_path),
                    "vworld_marked_image_path": str(marked_vworld_path),
                    "original_image_path": extraction_data['image_info']['path'],
                    "validation": validation
                }
                
                self.save_artifact(bbox_result, f"{self.image_name}_4_bbox_result.json", 'stage1_phase4')
                
                return bbox_result
            else:
                reason = validation.get('reasoning', 'Unknown') if validation else 'Validation failed'
                print(f"      ✗ Not validated: {reason}")
        
        print(f"\n    ✗ No match after {tested} attempts")
        return None
    
    def resolve_duplicates_for_combo(self, entries: List[Dict], image_info: Dict) -> List[Dict]:
        """Resolve duplicates by proximity"""
        transformer = Transformer.from_crs(CRS("EPSG:5179"), CRS("EPSG:4326"), always_xy=True)
        
        pois = []
        
        for entry in entries:
            coords_list = entry['gps_coords_epsg5179']
            
            if len(coords_list) == 1:
                lon, lat = transformer.transform(coords_list[0]['x'], coords_list[0]['y'])
                pois.append({
                    'index': entry['index'],
                    'name': entry['place_name'],
                    'pixel_x': entry['pixel_center']['x'],
                    'pixel_y': entry['pixel_center']['y'],
                    'gps_lon': lon,
                    'gps_lat': lat,
                    'gps_x_epsg5179': coords_list[0]['x'],
                    'gps_y_epsg5179': coords_list[0]['y']
                })
            else:
                best_match = None
                min_total_dist = float('inf')
                
                for candidate in coords_list:
                    lon, lat = transformer.transform(candidate['x'], candidate['y'])
                    
                    total_dist = 0
                    for other_poi in pois:
                        dist = great_circle((lat, lon), (other_poi['gps_lat'], other_poi['gps_lon'])).meters
                        total_dist += dist
                    
                    if total_dist < min_total_dist:
                        min_total_dist = total_dist
                        best_match = candidate
                
                if best_match:
                    lon, lat = transformer.transform(best_match['x'], best_match['y'])
                    pois.append({
                        'index': entry['index'],
                        'name': entry['place_name'],
                        'pixel_x': entry['pixel_center']['x'],
                        'pixel_y': entry['pixel_center']['y'],
                        'gps_lon': lon,
                        'gps_lat': lat,
                        'gps_x_epsg5179': best_match['x'],
                        'gps_y_epsg5179': best_match['y']
                    })
        
        return pois
    
    def calculate_scale_and_center_gps(self, image_size: Tuple[int, int],
                                        pois_with_coords: List[Dict]) -> Tuple[Optional[Tuple], Optional[float]]:
        """Calculate scale and center GPS"""
        if len(pois_with_coords) < 2:
            return None, None
        
        min_scale = None
        best_pair = None
        
        for i in range(len(pois_with_coords)):
            for j in range(i+1, len(pois_with_coords)):
                p1 = pois_with_coords[i]
                p2 = pois_with_coords[j]
                
                pixel_dist = math.sqrt((p1['pixel_x'] - p2['pixel_x'])**2 +
                                      (p1['pixel_y'] - p2['pixel_y'])**2)
                
                if pixel_dist == 0:
                    continue
                
                gps_dist_meters = great_circle(
                    (p1['gps_lat'], p1['gps_lon']),
                    (p2['gps_lat'], p2['gps_lon'])
                ).meters
                
                scale = gps_dist_meters / pixel_dist
                
                if min_scale is None or scale < min_scale:
                    min_scale = scale
                    best_pair = (p1, p2)
        
        if best_pair is None:
            return None, None
        
        p1, p2 = best_pair
        pixel_dist = math.sqrt((p1['pixel_x'] - p2['pixel_x'])**2 +
                              (p1['pixel_y'] - p2['pixel_y'])**2)
        gps_dist_meters = great_circle(
            (p1['gps_lat'], p1['gps_lon']),
            (p2['gps_lat'], p2['gps_lon'])
        ).meters
        meters_per_pixel = gps_dist_meters / pixel_dist
        
        img_center_x = image_size[0] / 2
        img_center_y = image_size[1] / 2
        
        vec_x = img_center_x - p1['pixel_x']
        vec_y = img_center_y - p1['pixel_y']
        
        vec_m_x = vec_x * meters_per_pixel
        vec_m_y = vec_y * meters_per_pixel
        
        m_per_deg_lat = 111132.954
        m_per_deg_lon = 111320 * math.cos(math.radians(p1['gps_lat']))
        
        offset_lon = vec_m_x / m_per_deg_lon
        offset_lat = -vec_m_y / m_per_deg_lat
        
        center_gps = (p1['gps_lon'] + offset_lon, p1['gps_lat'] + offset_lat)
        
        return center_gps, meters_per_pixel
    
    def calculate_zoom_level(self, center_lat: float, meters_per_pixel: float,
                            image_width: int) -> int:
        """Calculate zoom level"""
        world_circumference_m = 2 * math.pi * 6378137 * math.cos(math.radians(center_lat))
        zoom_float = math.log2(world_circumference_m / (256 * meters_per_pixel))
        zoom = round(zoom_float)
        zoom = max(7, min(18, zoom))
        return zoom
    
    def calculate_bbox_corners(self, center: Tuple[float, float], zoom: int, 
                              size: Tuple[int, int]) -> Dict:
        """Calculate GPS coordinates of bounding box corners"""
        # VWorld uses Web Mercator projection
        # At a given zoom level, the resolution (meters per pixel) at latitude is:
        # resolution = (156543.04 * cos(lat)) / (2^zoom)
        
        center_lon, center_lat = center
        width_px, height_px = size
        
        # Calculate resolution at center latitude
        resolution = (156543.04 * math.cos(math.radians(center_lat))) / (2 ** zoom)
        
        # Calculate offsets in meters
        half_width_m = (width_px / 2) * resolution
        half_height_m = (height_px / 2) * resolution
        
        # Convert meter offsets to degree offsets
        m_per_deg_lat = 111132.954
        m_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
        
        delta_lon = half_width_m / m_per_deg_lon
        delta_lat = half_height_m / m_per_deg_lat
        
        # Calculate corners
        corners = {
            "top_left": {
                "lon": center_lon - delta_lon,
                "lat": center_lat + delta_lat
            },
            "top_right": {
                "lon": center_lon + delta_lon,
                "lat": center_lat + delta_lat
            },
            "bottom_right": {
                "lon": center_lon + delta_lon,
                "lat": center_lat - delta_lat
            },
            "bottom_left": {
                "lon": center_lon - delta_lon,
                "lat": center_lat - delta_lat
            }
        }
        
        return corners
    
    def save_vworld_image(self, center: Tuple[float, float], zoom: int,
                         size: Tuple[int, int], output_path: Path) -> bool:
        """Fetch VWorld image"""
        url = "http://api.vworld.kr/req/image"
        
        params = {
            "SERVICE": "image",
            "REQUEST": "getmap",
            "KEY": self.vworld_api_key,
            "FORMAT": "png",
            "CRS": "epsg:4326",
            "CENTER": f"{center[0]},{center[1]}",
            "ZOOM": zoom,
            "SIZE": f"{size[0]},{size[1]}"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            ctype = response.headers.get('Content-Type', '')
            if 'image' in ctype:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                return False
                
        except Exception as e:
            print(f"      VWorld error: {e}")
            return False
    
    def validate_with_llm(self, original_image: str, vworld_image: str) -> Optional[Dict]:
        """Validate with LLM"""
        try:
            img1 = Image.open(original_image)
            img2 = Image.open(vworld_image)
            
            h = min(img1.height, img2.height, 800)
            w1 = int(img1.width * h / img1.height)
            w2 = int(img2.width * h / img2.height)
            
            img1 = img1.resize((w1, h))
            img2 = img2.resize((w2, h))
            
            comparison = Image.new('RGB', (w1 + w2 + 20, h), color='white')
            comparison.paste(img1, (0, 0))
            comparison.paste(img2, (w1 + 20, 0))
            
            comp_path = self.dirs['vworld_images'] / f"{self.image_name}_comparison.png"
            comparison.save(comp_path)
            
            prompt = """Compare these two maps (left: original, right: VWorld reference map).

Do they show the SAME geographic region? Check:
- Overall road/path patterns and layout
- Relative positions of landmarks
- Geographic features (rivers, mountains, etc.)
- General area structure

Respond ONLY with this JSON (no other text):
{
    "matches": "yes" or "no",
    "confidence": <0-100>,
    "reasoning": "<brief explanation>"
}"""
            
            response = self.llm.inference(str(comp_path), prompt, max_tokens=256)
            
            if not response:
                return {"matches": "error", "confidence": 0, "reasoning": "LLM failed"}
            
            validation = extract_json_from_text(response, debug=True)
            
            if validation:
                return validation
            else:
                return {"matches": "error", "confidence": 0, "reasoning": "Parse failed"}
                
        except Exception as e:
            return {"matches": "error", "confidence": 0, "reasoning": str(e)}
    
    # ========================================================================
    # STAGE 2A: AFFINE TRANSFORMATION & MAP ALIGNMENT
    # ========================================================================
    
    def stage2_affine_transformation(self, bbox_result: Dict) -> Dict:
        """
        Stage 2A: Affine transformation to align input map with VWorld map
        
        This stage:
        1. Uses the 3 POIs (from bbox extraction) as correspondence points
        2. Calculates affine transformation matrix
        3. Warps input map to align with VWorld map
        4. Creates overlapped visualization
        5. Generates coordinate transformation function
        
        Returns:
            Dictionary containing transformation results and overlapped image
        """
        self.log_step("STAGE 2A - Affine Transformation & Map Alignment")
        
        print(f"\n  Input map: {bbox_result['original_image_path']}")
        print(f"  VWorld reference: {bbox_result['vworld_image_path']}")
        
        # Extract POI information
        pois = bbox_result['confirmed_pois']
        center_gps = bbox_result['center_gps']
        zoom = bbox_result['zoom_level']
        vworld_size = (bbox_result['bbox_size']['width'], bbox_result['bbox_size']['height'])
        
        print(f"\n  Using {len(pois)} POIs for affine transformation:")
        for i, poi in enumerate(pois, 1):
            print(f"    {i}. Index {poi['index']}: {poi['name']}")
            print(f"       Input pixel: ({poi['pixel_x']:.1f}, {poi['pixel_y']:.1f})")
            print(f"       GPS: ({poi['gps_lon']:.6f}, {poi['gps_lat']:.6f})")
        
        # Step 1: Convert POI GPS coordinates to VWorld image pixel coordinates
        print(f"\n  [Step 1] Converting GPS to VWorld pixel coordinates...")
        vworld_poi_pixels = []
        
        for poi in pois:
            vworld_x, vworld_y = self.gps_to_vworld_pixel(
                poi['gps_lon'],
                poi['gps_lat'],
                center_gps['lon'],
                center_gps['lat'],
                zoom,
                vworld_size
            )
            vworld_poi_pixels.append({
                'index': poi['index'],
                'name': poi['name'],
                'x': vworld_x,
                'y': vworld_y
            })
            print(f"    POI {poi['index']}: VWorld pixel ({vworld_x:.1f}, {vworld_y:.1f})")
        
        # Step 2: Prepare correspondence points for affine transformation
        print(f"\n  [Step 2] Preparing correspondence points...")
        
        # Source points (input map pixels)
        src_points = np.float32([
            [pois[0]['pixel_x'], pois[0]['pixel_y']],
            [pois[1]['pixel_x'], pois[1]['pixel_y']],
            [pois[2]['pixel_x'], pois[2]['pixel_y']]
        ])
        
        # Destination points (VWorld map pixels)
        dst_points = np.float32([
            [vworld_poi_pixels[0]['x'], vworld_poi_pixels[0]['y']],
            [vworld_poi_pixels[1]['x'], vworld_poi_pixels[1]['y']],
            [vworld_poi_pixels[2]['x'], vworld_poi_pixels[2]['y']]
        ])
        
        print(f"    Source points (input map):")
        for i, pt in enumerate(src_points):
            print(f"      POI {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
        
        print(f"    Destination points (VWorld map):")
        for i, pt in enumerate(dst_points):
            print(f"      POI {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})")
        
        # Step 3: Calculate affine transformation matrix
        print(f"\n  [Step 3] Calculating affine transformation matrix...")
        
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        print(f"    Affine matrix:")
        print(f"      [{affine_matrix[0][0]:8.4f}, {affine_matrix[0][1]:8.4f}, {affine_matrix[0][2]:8.4f}]")
        print(f"      [{affine_matrix[1][0]:8.4f}, {affine_matrix[1][1]:8.4f}, {affine_matrix[1][2]:8.4f}]")
        
        # Step 4: Load images
        print(f"\n  [Step 4] Loading images...")
        
        input_img = cv2.imread(bbox_result['original_image_path'])
        vworld_img = cv2.imread(bbox_result['vworld_image_path'])
        
        if input_img is None:
            raise ValueError(f"Could not load input image: {bbox_result['original_image_path']}")
        if vworld_img is None:
            raise ValueError(f"Could not load VWorld image: {bbox_result['vworld_image_path']}")
        
        print(f"    Input map size: {input_img.shape[1]}x{input_img.shape[0]}")
        print(f"    VWorld map size: {vworld_img.shape[1]}x{vworld_img.shape[0]}")
        
        # Step 5: Apply affine transformation to warp input map
        print(f"\n  [Step 5] Applying affine transformation...")
        
        warped_input = cv2.warpAffine(
            input_img,
            affine_matrix,
            (vworld_img.shape[1], vworld_img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        print(f"    ✓ Warped input map to VWorld dimensions")
        
        # Step 6: Create mask for blending
        print(f"\n  [Step 6] Creating overlay mask...")
        
        # Create mask where warped input has content (non-black pixels)
        gray_warped = cv2.cvtColor(warped_input, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert mask to 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Step 7: Create overlapped visualization
        print(f"\n  [Step 7] Creating overlapped visualization...")
        
        # Blend: semi-transparent warped input over VWorld
        alpha = 0.6  # Transparency of warped input
        
        overlapped = vworld_img.copy()
        overlapped = overlapped * (1 - mask_3ch * alpha) + warped_input * mask_3ch * alpha
        overlapped = overlapped.astype(np.uint8)
        
        # Draw POI markers on overlapped image
        for vworld_poi in vworld_poi_pixels:
            x, y = int(vworld_poi['x']), int(vworld_poi['y'])
            # Draw circle
            cv2.circle(overlapped, (x, y), 8, (0, 255, 0), 2)
            # Draw cross
            cv2.drawMarker(overlapped, (x, y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 16, 2)
        
        # Save overlapped image
        overlap_path = self.dirs['stage2'] / f"{self.image_name}_affine_overlapped.png"
        cv2.imwrite(str(overlap_path), overlapped)
        print(f"    ✓ Saved overlapped image: {overlap_path}")
        
        # Also save warped input alone for reference
        warped_path = self.dirs['stage2'] / f"{self.image_name}_affine_warped.png"
        cv2.imwrite(str(warped_path), warped_input)
        print(f"    ✓ Saved warped input: {warped_path}")
        
        # Step 8: Create side-by-side comparison
        print(f"\n  [Step 8] Creating comparison visualization...")
        
        # Resize for consistent comparison
        h = min(vworld_img.shape[0], 800)
        w = int(vworld_img.shape[1] * h / vworld_img.shape[0])
        
        vworld_resized = cv2.resize(vworld_img, (w, h))
        overlapped_resized = cv2.resize(overlapped, (w, h))
        
        comparison = np.hstack([vworld_resized, overlapped_resized])
        
        comparison_path = self.dirs['stage2'] / f"{self.image_name}_affine_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"    ✓ Saved comparison: {comparison_path}")
        
        # Step 9: Package transformation results
        print(f"\n  [Step 9] Packaging transformation results...")
        
        transformation_result = {
            "stage": "2A",
            "method": "affine_transformation",
            "description": "Affine transformation using POI correspondences",
            "timestamp": datetime.now().isoformat(),
            
            # Original data
            "original_image": bbox_result['original_image_path'],
            "vworld_image": bbox_result['vworld_image_path'],
            "bbox_result_reference": f"{self.image_name}_4_bbox_result.json",
            
            # POI correspondence
            "correspondence_points": [
                {
                    "poi_index": poi['index'],
                    "poi_name": poi['name'],
                    "input_pixel": {"x": poi['pixel_x'], "y": poi['pixel_y']},
                    "gps": {"lon": poi['gps_lon'], "lat": poi['gps_lat']},
                    "vworld_pixel": {"x": vworld_poi_pixels[i]['x'], 
                                    "y": vworld_poi_pixels[i]['y']}
                }
                for i, poi in enumerate(pois)
            ],
            
            # Transformation matrix
            "affine_matrix": {
                "matrix": affine_matrix.tolist(),
                "description": "2x3 affine transformation matrix [a, b, tx; c, d, ty]"
            },
            
            # Output paths
            "outputs": {
                "overlapped_image": str(overlap_path),
                "warped_input_image": str(warped_path),
                "comparison_image": str(comparison_path)
            },
            
            # Image dimensions
            "dimensions": {
                "input_image": {
                    "width": input_img.shape[1],
                    "height": input_img.shape[0]
                },
                "vworld_image": {
                    "width": vworld_img.shape[1],
                    "height": vworld_img.shape[0]
                }
            },
            
            # VWorld parameters
            "vworld_params": {
                "center_gps": center_gps,
                "zoom_level": zoom,
                "size": vworld_size
            }
        }
        
        # Step 10: Create coordinate transformation function info
        print(f"\n  [Step 10] Creating coordinate transformation mapping...")
        
        # Test transformation on image corners and center
        test_points = {
            "top_left": [0, 0],
            "top_right": [input_img.shape[1], 0],
            "bottom_left": [0, input_img.shape[0]],
            "bottom_right": [input_img.shape[1], input_img.shape[0]],
            "center": [input_img.shape[1]/2, input_img.shape[0]/2]
        }
        
        transformed_test_points = {}
        for name, pt in test_points.items():
            # Apply affine transformation
            pt_array = np.array([[pt[0], pt[1]]], dtype=np.float32)
            transformed = cv2.transform(pt_array.reshape(-1, 1, 2), affine_matrix)
            tx, ty = transformed[0][0]
            
            # Convert to GPS
            gps_lon, gps_lat = self.vworld_pixel_to_gps(
                tx, ty,
                center_gps['lon'], center_gps['lat'],
                zoom, vworld_size
            )
            
            transformed_test_points[name] = {
                "input_pixel": {"x": float(pt[0]), "y": float(pt[1])},
                "vworld_pixel": {"x": float(tx), "y": float(ty)},
                "gps": {"lon": float(gps_lon), "lat": float(gps_lat)}
            }
        
        transformation_result["coordinate_mapping"] = {
            "description": "Test points showing input → VWorld → GPS transformation",
            "test_points": transformed_test_points,
            "transformation_formula": {
                "description": "Apply affine matrix, then convert VWorld pixel to GPS",
                "step1": "vworld_pixel = affine_matrix @ [input_x, input_y, 1]",
                "step2": "gps = vworld_pixel_to_gps(vworld_pixel, center, zoom, size)"
            }
        }
        
        # Save transformation result
        self.save_artifact(
            transformation_result,
            f"{self.image_name}_stage2a_affine_transformation.json",
            'stage2'
        )
        
        print(f"\n✓ Stage 2A Complete!")
        print(f"  Overlapped image: {overlap_path}")
        print(f"  Transformation data: {self.dirs['stage2']}/{self.image_name}_stage2a_affine_transformation.json")
        
        return transformation_result
    
    # ========================================================================
    # STAGE 2B: ADVANCED FEATURE MATCHING & MAP ALIGNMENT
    # ========================================================================
    
    def stage2_feature_matching(self, bbox_result: Dict) -> Dict:
        """
        Stage 2B: Advanced feature matching to align input map with VWorld map
        
        This stage:
        1. Extracts multi-scale features (road, water, edges, gradients, orientation)
        2. Uses sliding window with composite similarity (ZNCC, Chamfer, Orientation)
        3. Finds best matching position
        4. Creates overlapped visualization
        5. Records matching position and all scores
        
        Returns:
            Dictionary containing matching results and overlapped image
        """
        self.log_step("STAGE 2B - Advanced Feature Matching & Map Alignment")
        
        print(f"\n  Input map: {bbox_result['original_image_path']}")
        print(f"  VWorld reference: {bbox_result['vworld_image_path']}")
        print(f"  Using advanced features: road mask, water mask, edges, gradients, orientation")
        print(f"  Using metrics: ZNCC, Chamfer distance, Orientation similarity")
        
        # Extract basic info
        center_gps = bbox_result['center_gps']
        zoom = bbox_result['zoom_level']
        vworld_size = (bbox_result['bbox_size']['width'], bbox_result['bbox_size']['height'])
        
        # Step 1: Load images
        print(f"\n  [Step 1] Loading images...")
        
        input_img = cv2.imread(bbox_result['original_image_path'])
        vworld_img = cv2.imread(bbox_result['vworld_image_path'])
        
        if input_img is None:
            raise ValueError(f"Could not load input image: {bbox_result['original_image_path']}")
        if vworld_img is None:
            raise ValueError(f"Could not load VWorld image: {bbox_result['vworld_image_path']}")
        
        print(f"    Input map size: {input_img.shape[1]}x{input_img.shape[0]}")
        print(f"    VWorld map size: {vworld_img.shape[1]}x{vworld_img.shape[0]}")
        
        # Step 2: Extract multi-scale features
        print(f"\n  [Step 2] Extracting multi-scale features...")
        
        print(f"    Processing input map...")
        input_features = extract_multi_scale_features(input_img)
        
        print(f"    Processing VWorld map...")
        vworld_features = extract_multi_scale_features(vworld_img)
        
        print(f"    ✓ Feature extraction complete")
        
        # Save feature visualizations
        print(f"\n  [Step 3] Saving feature visualizations...")
        
        feature_types = ['grayscale', 'road_mask', 'water_mask', 'canny_edges', 'sobel_magnitude']
        for feat_type in feature_types:
            # Input
            feat_vis = (input_features[feat_type] * 255).astype(np.uint8)
            if feat_type == 'orientation':
                # Special handling for orientation (colorize)
                feat_vis = ((input_features[feat_type] + np.pi) / (2*np.pi) * 255).astype(np.uint8)
                feat_vis = cv2.applyColorMap(feat_vis, cv2.COLORMAP_HSV)
            
            feat_path = self.dirs['stage2_features'] / f"{self.image_name}_input_{feat_type}.png"
            cv2.imwrite(str(feat_path), feat_vis)
            
            # VWorld
            feat_vis = (vworld_features[feat_type] * 255).astype(np.uint8)
            if feat_type == 'orientation':
                feat_vis = ((vworld_features[feat_type] + np.pi) / (2*np.pi) * 255).astype(np.uint8)
                feat_vis = cv2.applyColorMap(feat_vis, cv2.COLORMAP_HSV)
            
            feat_path = self.dirs['stage2_features'] / f"{self.image_name}_vworld_{feat_type}.png"
            cv2.imwrite(str(feat_path), feat_vis)
        
        print(f"    ✓ Saved feature visualizations")
        
        # Step 4: Advanced sliding window matching
        print(f"\n  [Step 4] Advanced sliding window matching...")
        
        # Determine step size based on image size
        step_size = max(5, min(input_img.shape[0], input_img.shape[1]) // 50)
        print(f"    Using step size: {step_size} pixels")
        
        best_pos, best_score, score_map, individual_scores = sliding_window_match_advanced(
            input_features,
            vworld_features,
            step_size=step_size
        )
        
        best_y, best_x = best_pos
        print(f"\n    ✓ Best match found:")
        print(f"      Position: ({best_x}, {best_y})")
        print(f"      Composite Score: {best_score:.4f}")
                
        # Save heatmap
        print(f"\n  [Step 5] Creating similarity heatmap...")

        # Normalize score map for visualization (handle NaN/inf gracefully)
        score_map_clean = score_map.copy()
        # Replace -inf with minimum valid value
        score_map_clean[np.isinf(score_map_clean) & (score_map_clean < 0)] = np.nanmin(score_map_clean[np.isfinite(score_map_clean)])
        # Replace inf with maximum valid value
        score_map_clean[np.isinf(score_map_clean) & (score_map_clean > 0)] = np.nanmax(score_map_clean[np.isfinite(score_map_clean)])
        # Replace NaN with minimum valid value
        score_map_clean[np.isnan(score_map_clean)] = np.nanmin(score_map_clean[np.isfinite(score_map_clean)])

        # Normalize
        if np.max(score_map_clean) > np.min(score_map_clean):
            score_map_norm = (score_map_clean - score_map_clean.min()) / (score_map_clean.max() - score_map_clean.min())
        else:
            # All values are the same, use zeros
            score_map_norm = np.zeros_like(score_map_clean)

        heatmap = (score_map_norm * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Mark best position
        cv2.circle(heatmap_colored, (best_x, best_y), 10, (255, 255, 255), 3)
        cv2.drawMarker(heatmap_colored, (best_x, best_y), (255, 255, 255),
                      cv2.MARKER_CROSS, 20, 3)
        
        heatmap_path = self.dirs['stage2'] / f"{self.image_name}_feature_heatmap.png"
        cv2.imwrite(str(heatmap_path), heatmap_colored)
        print(f"    ✓ Saved heatmap: {heatmap_path}")
        
        # Step 6: Create overlapped visualization
        print(f"\n  [Step 6] Creating overlapped visualization...")
        
        # Extract matching region from VWorld
        input_h, input_w = input_img.shape[:2]
        matched_region = vworld_img[best_y:best_y+input_h, best_x:best_x+input_w].copy()
        
        # Blend input and matched region
        alpha = 0.6
        blended = cv2.addWeighted(matched_region, 1-alpha, input_img, alpha, 0)
        
        # Create full overlapped image (place blended result on VWorld)
        overlapped = vworld_img.copy()
        overlapped[best_y:best_y+input_h, best_x:best_x+input_w] = blended
        
        # Draw rectangle around matched region
        cv2.rectangle(overlapped, (best_x, best_y), 
                     (best_x + input_w, best_y + input_h),
                     (0, 255, 0), 3)
        
        # Add score text
        score_text = f"Score: {best_score:.3f}"
        cv2.putText(overlapped, score_text, (best_x + 10, best_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Save overlapped image
        overlap_path = self.dirs['stage2'] / f"{self.image_name}_feature_overlapped.png"
        cv2.imwrite(str(overlap_path), overlapped)
        print(f"    ✓ Saved overlapped image: {overlap_path}")
        
        # Save blended region alone
        blended_path = self.dirs['stage2'] / f"{self.image_name}_feature_blended.png"
        cv2.imwrite(str(blended_path), blended)
        print(f"    ✓ Saved blended region: {blended_path}")
        
        # Step 7: Create side-by-side comparison
        print(f"\n  [Step 7] Creating comparison visualization...")
        
        # Resize for display
        h = min(vworld_img.shape[0], 800)
        w = int(vworld_img.shape[1] * h / vworld_img.shape[0])
        
        vworld_resized = cv2.resize(vworld_img, (w, h))
        overlapped_resized = cv2.resize(overlapped, (w, h))
        
        comparison = np.hstack([vworld_resized, overlapped_resized])
        
        comparison_path = self.dirs['stage2'] / f"{self.image_name}_feature_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"    ✓ Saved comparison: {comparison_path}")
        
        # Step 8: Calculate GPS coordinates of matched region corners
        print(f"\n  [Step 8] Calculating GPS coordinates...")
        
        # Convert pixel positions to GPS
        corners_gps = {}
        corner_positions = {
            "top_left": (best_x, best_y),
            "top_right": (best_x + input_w, best_y),
            "bottom_right": (best_x + input_w, best_y + input_h),
            "bottom_left": (best_x, best_y + input_h)
        }
        
        for corner_name, (px, py) in corner_positions.items():
            lon, lat = self.vworld_pixel_to_gps(
                px, py,
                center_gps['lon'], center_gps['lat'],
                zoom, vworld_size
            )
            corners_gps[corner_name] = {"lon": lon, "lat": lat}
            print(f"    {corner_name}: ({lon:.6f}, {lat:.6f})")
        
        # Calculate center of matched region
        center_x = best_x + input_w / 2
        center_y = best_y + input_h / 2
        center_lon, center_lat = self.vworld_pixel_to_gps(
            center_x, center_y,
            center_gps['lon'], center_gps['lat'],
            zoom, vworld_size
        )
        
        # Step 9: Package matching results
        print(f"\n  [Step 9] Packaging matching results...")
        
        matching_result = {
            "stage": "2B",
            "method": "feature_matching",
            "description": "Advanced feature-based sliding window matching with ZNCC, Chamfer, and Orientation",
            "timestamp": datetime.now().isoformat(),
            
            # Original data
            "original_image": bbox_result['original_image_path'],
            "vworld_image": bbox_result['vworld_image_path'],
            "bbox_result_reference": f"{self.image_name}_4_bbox_result.json",
            
            # Matching results
            "best_match": {
                "position": {"x": int(best_x), "y": int(best_y)},
                "composite_score": float(best_score),
                "individual_scores": {k: float(v) for k, v in individual_scores.items()},
                "metrics_used": ["zncc_grayscale", "zncc_road", "zncc_water", 
                               "zncc_sobel", "chamfer_canny", "orientation"]
            },
            
            # Matched region
            "matched_region": {
                "vworld_pixels": {
                    "top_left": {"x": int(best_x), "y": int(best_y)},
                    "top_right": {"x": int(best_x + input_w), "y": int(best_y)},
                    "bottom_right": {"x": int(best_x + input_w), "y": int(best_y + input_h)},
                    "bottom_left": {"x": int(best_x), "y": int(best_y + input_h)},
                    "center": {"x": float(center_x), "y": float(center_y)}
                },
                "gps_coordinates": corners_gps,
                "center_gps": {"lon": center_lon, "lat": center_lat}
            },
            
            # Output paths
            "outputs": {
                "overlapped_image": str(overlap_path),
                "blended_region": str(blended_path),
                "comparison_image": str(comparison_path),
                "heatmap": str(heatmap_path)
            },
            
            # Image dimensions
            "dimensions": {
                "input_image": {
                    "width": input_img.shape[1],
                    "height": input_img.shape[0]
                },
                "vworld_image": {
                    "width": vworld_img.shape[1],
                    "height": vworld_img.shape[0]
                }
            },
            
            # VWorld parameters
            "vworld_params": {
                "center_gps": center_gps,
                "zoom_level": zoom,
                "size": vworld_size
            },
            
            # Feature extraction parameters
            "feature_params": {
                "features_used": ["grayscale", "road_mask", "water_mask", 
                                "canny_edges", "sobel_magnitude", "orientation"],
                "step_size": step_size
            }
        }
        
        # Step 10: Create coordinate mapping
        print(f"\n  [Step 10] Creating coordinate transformation mapping...")
        
        # Test transformation on input image corners and center
        test_points = {
            "top_left": [0, 0],
            "top_right": [input_img.shape[1], 0],
            "bottom_left": [0, input_img.shape[0]],
            "bottom_right": [input_img.shape[1], input_img.shape[0]],
            "center": [input_img.shape[1]/2, input_img.shape[0]/2]
        }
        
        transformed_test_points = {}
        for name, pt in test_points.items():
            # Map to VWorld pixel
            vworld_x = best_x + pt[0]
            vworld_y = best_y + pt[1]
            
            # Convert to GPS
            gps_lon, gps_lat = self.vworld_pixel_to_gps(
                vworld_x, vworld_y,
                center_gps['lon'], center_gps['lat'],
                zoom, vworld_size
            )
            
            transformed_test_points[name] = {
                "input_pixel": {"x": float(pt[0]), "y": float(pt[1])},
                "vworld_pixel": {"x": float(vworld_x), "y": float(vworld_y)},
                "gps": {"lon": float(gps_lon), "lat": float(gps_lat)}
            }
        
        matching_result["coordinate_mapping"] = {
            "description": "Test points showing input → VWorld → GPS transformation",
            "test_points": transformed_test_points,
            "transformation_formula": {
                "description": "Simple translation: add best_match position offset",
                "step1": f"vworld_pixel = input_pixel + ({best_x}, {best_y})",
                "step2": "gps = vworld_pixel_to_gps(vworld_pixel, center, zoom, size)"
            }
        }
        
        # Save matching result
        self.save_artifact(
            matching_result,
            f"{self.image_name}_stage2b_feature_matching.json",
            'stage2'
        )
        
        print(f"\n✓ Stage 2B Complete!")
        print(f"  Method: Advanced Feature Matching")
        print(f"  Features: Road mask, Water mask, Edges (Canny/Sobel), Gradients, Orientation")
        print(f"  Metrics: ZNCC, Chamfer distance, Orientation similarity")
        print(f"  Overlapped image: {overlap_path}")
        print(f"  Matching data: {self.dirs['stage2']}/{self.image_name}_stage2b_feature_matching.json")
        
        return matching_result
    
    # ========================================================================
    # COORDINATE TRANSFORMATION UTILITIES
    # ========================================================================
    
    def gps_to_vworld_pixel(self, lon: float, lat: float, 
                           center_lon: float, center_lat: float,
                           zoom: int, size: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert GPS coordinates to VWorld image pixel coordinates
        
        Args:
            lon, lat: GPS coordinates to convert
            center_lon, center_lat: Center GPS of VWorld image
            zoom: VWorld zoom level
            size: VWorld image size (width, height)
            
        Returns:
            (pixel_x, pixel_y) in VWorld image
        """
        # Calculate resolution at center latitude
        resolution = (156543.04 * math.cos(math.radians(center_lat))) / (2 ** zoom)
        
        # Calculate GPS offsets from center
        m_per_deg_lat = 111132.954
        m_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
        
        offset_lon_deg = lon - center_lon
        offset_lat_deg = lat - center_lat
        
        # Convert to meters
        offset_x_m = offset_lon_deg * m_per_deg_lon
        offset_y_m = offset_lat_deg * m_per_deg_lat
        
        # Convert to pixels (y is inverted)
        offset_x_px = offset_x_m / resolution
        offset_y_px = -offset_y_m / resolution
        
        # Add to center pixel
        center_x = size[0] / 2
        center_y = size[1] / 2
        
        pixel_x = center_x + offset_x_px
        pixel_y = center_y + offset_y_px
        
        return pixel_x, pixel_y
    
    def vworld_pixel_to_gps(self, pixel_x: float, pixel_y: float,
                           center_lon: float, center_lat: float,
                           zoom: int, size: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert VWorld image pixel coordinates to GPS coordinates
        
        Args:
            pixel_x, pixel_y: Pixel coordinates in VWorld image
            center_lon, center_lat: Center GPS of VWorld image
            zoom: VWorld zoom level
            size: VWorld image size (width, height)
            
        Returns:
            (lon, lat) GPS coordinates
        """
        # Calculate resolution at center latitude
        resolution = (156543.04 * math.cos(math.radians(center_lat))) / (2 ** zoom)
        
        # Calculate pixel offsets from center
        center_x = size[0] / 2
        center_y = size[1] / 2
        
        offset_x_px = pixel_x - center_x
        offset_y_px = pixel_y - center_y
        
        # Convert to meters (y is inverted)
        offset_x_m = offset_x_px * resolution
        offset_y_m = -offset_y_px * resolution
        
        # Convert to degrees
        m_per_deg_lat = 111132.954
        m_per_deg_lon = 111320 * math.cos(math.radians(center_lat))
        
        offset_lon_deg = offset_x_m / m_per_deg_lon
        offset_lat_deg = offset_y_m / m_per_deg_lat
        
        # Add to center GPS
        lon = center_lon + offset_lon_deg
        lat = center_lat + offset_lat_deg
        
        return lon, lat
    
    def transform_input_pixel_to_gps(self, input_x: float, input_y: float,
                                     transformation_data: Dict) -> Tuple[float, float]:
        """
        Transform input map pixel to GPS coordinates (works for both affine and feature methods)
        
        Args:
            input_x, input_y: Pixel coordinates in input image
            transformation_data: Result from stage 2 (affine or feature)
            
        Returns:
            (lon, lat) GPS coordinates
        """
        method = transformation_data.get('method', 'affine_transformation')
        
        if method == 'affine_transformation':
            # Use affine matrix
            affine_matrix = np.array(transformation_data['affine_matrix']['matrix'])
            
            # Apply transformation
            pt = np.array([[input_x, input_y]], dtype=np.float32)
            transformed = cv2.transform(pt.reshape(-1, 1, 2), affine_matrix)
            vworld_x, vworld_y = transformed[0][0]
            
        elif method == 'feature_matching':
            # Simple translation
            best_x = transformation_data['best_match']['position']['x']
            best_y = transformation_data['best_match']['position']['y']
            
            vworld_x = input_x + best_x
            vworld_y = input_y + best_y
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert VWorld pixel to GPS
        vworld_params = transformation_data['vworld_params']
        center_gps = vworld_params['center_gps']
        zoom = vworld_params['zoom_level']
        size = (vworld_params['size'][0], vworld_params['size'][1])
        
        lon, lat = self.vworld_pixel_to_gps(
            vworld_x, vworld_y,
            center_gps['lon'], center_gps['lat'],
            zoom, size
        )
        
        return lon, lat

    # ========================================================================
    # STAGE 3: HIGHLIGHT EXTRACTION
    # ========================================================================

    def stage3_highlight_extraction(self, image_path: str, checkpoint_path: str) -> Dict:
        """
        Stage 3: Extract highlight from input map
        
        Args:
            image_path: Input map image path
            checkpoint_path: Path to trained highlight extraction model
            
        Returns:
            Dictionary containing extraction results
        """
        self.log_step("STAGE 3 - Highlight Extraction from Input Map")
        
        print(f"\n  Input map: {image_path}")
        print(f"  Model checkpoint: {checkpoint_path}")
        
        # Extract highlight mask
        mask_result = extract_highlight_mask(
            image_path,
            self.dirs['stage3'],
            checkpoint_path,
            threshold=0.1,
            image_size=512
        )
        
        # Skeletonize
        skeleton_result = skeletonize_highlight(
            mask_result['mask'],
            self.dirs['stage3'],
            self.image_name
        )
        
        highlight_result = {
            "stage": "3",
            "description": "Highlight extraction and skeletonization",
            "timestamp": datetime.now().isoformat(),
            "input_image": image_path,
            "model_checkpoint": checkpoint_path,
            "threshold": 0.1,
            "outputs": {
                "raw_mask": mask_result['mask_path'],
                "overlay": mask_result['overlay_path'],
                "skeleton": skeleton_result['skeleton_path'],
                "skeleton_overlay": skeleton_result['skeleton_overlay_path']
            },
            "skeleton_pixels": skeleton_result['skeleton_pixels'].tolist(),
            "num_skeleton_pixels": len(skeleton_result['skeleton_pixels'])
            # Removed "mask": mask_result['mask'] - it's a numpy array and can't be JSON serialized
        }
        
        # Save artifact (without mask)
        self.save_artifact(
            highlight_result,
            f"{self.image_name}_stage3_highlight_extraction.json",
            'stage3'
        )
        
        # Add mask back to result for passing to stage 4 (not saved in JSON)
        highlight_result['mask'] = mask_result['mask']
        
        print(f"\n✓ Stage 3 Complete!")
        print(f"  Extracted {len(skeleton_result['skeleton_pixels'])} skeleton pixels")
        print(f"  Outputs saved to: {self.dirs['stage3']}")
        
        return highlight_result
 
    # ========================================================================
    # STAGE 4: MAP HIGHLIGHT TO VWORLD BOUNDING BOX
    # ========================================================================
    
    def stage4_map_highlight_to_vworld(self, highlight_result: Dict, 
                                      transformation_result: Dict) -> Dict:
        """
        Stage 4: Map extracted highlight to VWorld bounding box
        
        Args:
            highlight_result: Result from Stage 3
            transformation_result: Result from Stage 2
            
        Returns:
            Dictionary containing mapped highlight
        """
        self.log_step("STAGE 4 - Map Highlight to VWorld Bounding Box")
        
        skeleton_pixels = np.array(highlight_result['skeleton_pixels'])
        print(f"\n  Mapping {len(skeleton_pixels)} skeleton pixels to VWorld...")
        
        method = transformation_result.get('method', 'affine_transformation')
        
        # Transform skeleton pixels
        if method == 'affine_transformation':
            print(f"  Using affine transformation...")
            affine_matrix = np.array(transformation_result['affine_matrix']['matrix'])
            
            # Apply affine transformation
            skeleton_pixels_homogeneous = np.column_stack([skeleton_pixels, np.ones(len(skeleton_pixels))])
            vworld_pixels = (affine_matrix @ skeleton_pixels_homogeneous.T).T
            
        elif method == 'feature_matching':
            print(f"  Using feature matching translation...")
            best_x = transformation_result['best_match']['position']['x']
            best_y = transformation_result['best_match']['position']['y']
            
            # Simple translation
            vworld_pixels = skeleton_pixels + np.array([best_x, best_y])
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"  ✓ Transformed to VWorld coordinates")
        
        # Load VWorld image
        vworld_img_path = transformation_result.get('vworld_image') or \
                         transformation_result.get('original_image')
        
        if not vworld_img_path:
            raise ValueError("VWorld image path not found in transformation result")
        
        vworld_img = cv2.imread(vworld_img_path)
        if vworld_img is None:
            raise ValueError(f"Could not load VWorld image: {vworld_img_path}")
        
        print(f"  VWorld image size: {vworld_img.shape[1]}x{vworld_img.shape[0]}")
        
        # Filter pixels within VWorld image bounds
        h, w = vworld_img.shape[:2]
        valid_mask = (vworld_pixels[:, 0] >= 0) & (vworld_pixels[:, 0] < w) & \
                     (vworld_pixels[:, 1] >= 0) & (vworld_pixels[:, 1] < h)
        
        vworld_pixels_valid = vworld_pixels[valid_mask]
        
        print(f"  Valid pixels in VWorld bounds: {len(vworld_pixels_valid)}/{len(vworld_pixels)}")
        
        # Save the initial mapped pixels (before any processing)
        initial_mapped_path = self.dirs['stage4'] / f"{self.image_name}_initial_mapped_pixels.npy"
        np.save(initial_mapped_path, vworld_pixels_valid)
        print(f"  ✓ Saved initial mapped pixels: {initial_mapped_path}")
        
        # Load input image for side-by-side visualization
        input_img = cv2.imread(highlight_result['input_image'])
        
        # Create visualization showing the ORIGINAL MASK (not skeleton) on input image
        print(f"\n  Creating initial highlight visualization (raw mask before skeleton)...")
        
        # Get the mask from highlight_result
        if 'mask' in highlight_result:
            initial_mask = highlight_result['mask']
        else:
            # Fallback: read from file
            initial_mask = cv2.imread(highlight_result['outputs']['raw_mask'], cv2.IMREAD_GRAYSCALE)
        
        # Create visualization with raw mask on input image
        input_with_mask = input_img.copy()
        input_with_mask[initial_mask > 0] = (0, 255, 0)  # Green highlight
        input_with_mask = cv2.addWeighted(input_img, 0.7, input_with_mask, 0.3, 0)
        
        # Create visualization with mapped pixels on VWorld
        vworld_with_pixels = vworld_img.copy()
        for px, py in vworld_pixels_valid:
            px, py = int(px), int(py)
            cv2.circle(vworld_with_pixels, (px, py), 1, (255, 0, 255), -1)  # Magenta
        
        # Resize both to same height for side-by-side
        h = min(input_with_mask.shape[0], vworld_with_pixels.shape[0], 800)
        w1 = int(input_with_mask.shape[1] * h / input_with_mask.shape[0])
        w2 = int(vworld_with_pixels.shape[1] * h / vworld_with_pixels.shape[0])
        
        input_resized = cv2.resize(input_with_mask, (w1, h))
        vworld_resized = cv2.resize(vworld_with_pixels, (w2, h))
        
        # Create side-by-side
        initial_sidebyside = np.hstack([input_resized, vworld_resized])
        initial_sidebyside_path = self.dirs['stage4'] / f"{self.image_name}_initial_highlight_sidebyside.png"
        cv2.imwrite(str(initial_sidebyside_path), initial_sidebyside)
        print(f"  ✓ Saved initial side-by-side: {initial_sidebyside_path}")
        
        # Also save individual images
        initial_input_path = self.dirs['stage4'] / f"{self.image_name}_initial_highlight_on_input.png"
        cv2.imwrite(str(initial_input_path), input_with_mask)
        
        initial_vworld_path = self.dirs['stage4'] / f"{self.image_name}_initial_highlight_on_vworld.png"
        cv2.imwrite(str(initial_vworld_path), vworld_with_pixels)
        
        # Draw final highlight (skeleton) on VWorld
        vworld_with_highlight = vworld_img.copy()
        
        for px, py in vworld_pixels_valid:
            px, py = int(px), int(py)
            cv2.circle(vworld_with_highlight, (px, py), 1, (0, 255, 0), -1)  # Green
        
        # Save
        highlight_on_vworld_path = self.dirs['stage4'] / f"{self.image_name}_highlight_on_vworld.png"
        cv2.imwrite(str(highlight_on_vworld_path), vworld_with_highlight)
        print(f"  ✓ Saved highlight on VWorld: {highlight_on_vworld_path}")
        
        mapping_result = {
            "stage": "4",
            "description": "Highlight mapped to VWorld bounding box",
            "timestamp": datetime.now().isoformat(),
            "transformation_method": method,
            "input_skeleton_pixels": len(skeleton_pixels),
            "vworld_mapped_pixels": len(vworld_pixels_valid),
            "vworld_pixels": vworld_pixels_valid.tolist(),
            "outputs": {
                "initial_mapped_pixels_npy": str(initial_mapped_path),
                "initial_highlight_on_input": str(initial_input_path),
                "initial_highlight_on_vworld": str(initial_vworld_path),
                "initial_sidebyside": str(initial_sidebyside_path),
                "highlight_on_vworld": str(highlight_on_vworld_path)
            }
        }
        
        self.save_artifact(
            mapping_result,
            f"{self.image_name}_stage4_highlight_mapping.json",
            'stage4'
        )
        
        print(f"\n✓ Stage 4 Complete!")
        print(f"  Mapped {len(vworld_pixels_valid)} highlight pixels to VWorld")
        print(f"  Initial mask visualization saved (before skeleton)")
        print(f"  Side-by-side comparison saved")
        
        return mapping_result
    
    # ========================================================================
    # STAGE 5: CALCULATE GPS COORDINATES OF HIGHLIGHT
    # ========================================================================
    
    def stage5_calculate_highlight_gps(self, mapping_result: Dict,
                                      transformation_result: Dict) -> Dict:
        """
        Stage 5: Calculate actual GPS coordinates of highlight polyline
        
        Args:
            mapping_result: Result from Stage 4
            transformation_result: Result from Stage 2 (for VWorld params)
            
        Returns:
            Dictionary containing GPS polyline and visualizations
        """
        self.log_step("STAGE 5 - Calculate GPS Coordinates of Highlight Polyline")
        
        vworld_pixels = np.array(mapping_result['vworld_pixels'])
        print(f"\n  Converting {len(vworld_pixels)} pixels to GPS coordinates...")
        
        # Get VWorld parameters
        vworld_params = transformation_result['vworld_params']
        center_gps = vworld_params['center_gps']
        zoom = vworld_params['zoom_level']
        size = (vworld_params['size'][0], vworld_params['size'][1])
        
        # Convert to GPS
        gps_coords = []
        for px, py in vworld_pixels:
            lon, lat = self.vworld_pixel_to_gps(
                px, py,
                center_gps['lon'], center_gps['lat'],
                zoom, size
            )
            gps_coords.append([lon, lat])
        
        gps_coords = np.array(gps_coords)
        print(f"  ✓ Converted to GPS coordinates")
        
        # Order points along polyline (simple nearest-neighbor ordering)
        print(f"  Ordering points along polyline...")
        ordered_indices = self._order_polyline_points(gps_coords)
        gps_polyline = gps_coords[ordered_indices]
        
        print(f"  ✓ Polyline ordered: {len(gps_polyline)} points")
        
        # Load VWorld image and input image
        vworld_img_path = transformation_result.get('vworld_image') or \
                         transformation_result.get('original_image')
        vworld_img = cv2.imread(vworld_img_path)
        
        # Load input image from mapping result or transformation result
        input_img_path = None
        if 'input_image' in mapping_result:
            input_img_path = mapping_result['input_image']
        else:
            # Try to get from transformation result
            input_img_path = transformation_result.get('original_image')
        
        if input_img_path:
            input_img = cv2.imread(input_img_path)
        else:
            input_img = None
        
        # Create visualizations
        print(f"\n  Creating visualizations...")
        
        # 1. Thin polyline visualization on VWorld
        vis_thin = vworld_img.copy()
        ordered_pixels = vworld_pixels[ordered_indices]
        
        for i in range(len(ordered_pixels) - 1):
            pt1 = tuple(ordered_pixels[i].astype(int))
            pt2 = tuple(ordered_pixels[i + 1].astype(int))
            cv2.line(vis_thin, pt1, pt2, (0, 255, 0), 2)
        
        vis_thin_path = self.dirs['stage5'] / f"{self.image_name}_polyline_thin.png"
        cv2.imwrite(str(vis_thin_path), vis_thin)
        print(f"  ✓ Saved thin polyline: {vis_thin_path}")
        
        # 2. Thick and smooth polyline visualization on VWorld
        print(f"  Smoothing polyline...")
        smoothed_pixels = smooth_polyline(ordered_pixels, smoothing=5.0, num_points=len(ordered_pixels))
        
        vis_thick = vworld_img.copy()
        for i in range(len(smoothed_pixels) - 1):
            pt1 = tuple(smoothed_pixels[i].astype(int))
            pt2 = tuple(smoothed_pixels[i + 1].astype(int))
            cv2.line(vis_thick, pt1, pt2, (0, 255, 0), 6)
        
        vis_thick_path = self.dirs['stage5'] / f"{self.image_name}_polyline_thick_smooth.png"
        cv2.imwrite(str(vis_thick_path), vis_thick)
        print(f"  ✓ Saved thick smooth polyline: {vis_thick_path}")
        
        # 3. Create side-by-side visualizations showing input and VWorld with polyline
        if input_img is not None:
            print(f"  Creating side-by-side visualizations...")
            
            # For input image, we need to draw the polyline in input coordinates
            # Get transformation method
            method = transformation_result.get('method', 'affine_transformation')
            
            # Transform ordered pixels back to input coordinates
            if method == 'affine_transformation':
                # Inverse affine transformation
                affine_matrix = np.array(transformation_result['affine_matrix']['matrix'])
                # Compute inverse
                A = affine_matrix[:, :2]
                b = affine_matrix[:, 2]
                A_inv = np.linalg.inv(A)
                
                input_pixels = []
                for vworld_pt in ordered_pixels:
                    input_pt = A_inv @ (vworld_pt - b)
                    input_pixels.append(input_pt)
                input_pixels = np.array(input_pixels)
                
            elif method == 'feature_matching':
                # Reverse translation
                best_x = transformation_result['best_match']['position']['x']
                best_y = transformation_result['best_match']['position']['y']
                input_pixels = ordered_pixels - np.array([best_x, best_y])
            
            # Draw thin polyline on input
            input_with_thin = input_img.copy()
            for i in range(len(input_pixels) - 1):
                pt1 = tuple(input_pixels[i].astype(int))
                pt2 = tuple(input_pixels[i + 1].astype(int))
                cv2.line(input_with_thin, pt1, pt2, (0, 255, 0), 2)
            
            # Draw thick smooth polyline on input
            smoothed_input_pixels = smooth_polyline(input_pixels, smoothing=5.0, num_points=len(input_pixels))
            input_with_thick = input_img.copy()
            for i in range(len(smoothed_input_pixels) - 1):
                pt1 = tuple(smoothed_input_pixels[i].astype(int))
                pt2 = tuple(smoothed_input_pixels[i + 1].astype(int))
                cv2.line(input_with_thick, pt1, pt2, (0, 255, 0), 6)
            
            # Resize for side-by-side (thin)
            h = min(input_with_thin.shape[0], vis_thin.shape[0], 800)
            w1 = int(input_with_thin.shape[1] * h / input_with_thin.shape[0])
            w2 = int(vis_thin.shape[1] * h / vis_thin.shape[0])
            
            input_thin_resized = cv2.resize(input_with_thin, (w1, h))
            vworld_thin_resized = cv2.resize(vis_thin, (w2, h))
            
            sidebyside_thin = np.hstack([input_thin_resized, vworld_thin_resized])
            sidebyside_thin_path = self.dirs['stage5'] / f"{self.image_name}_polyline_thin_sidebyside.png"
            cv2.imwrite(str(sidebyside_thin_path), sidebyside_thin)
            print(f"  ✓ Saved thin side-by-side: {sidebyside_thin_path}")
            
            # Resize for side-by-side (thick)
            input_thick_resized = cv2.resize(input_with_thick, (w1, h))
            vworld_thick_resized = cv2.resize(vis_thick, (w2, h))
            
            sidebyside_thick = np.hstack([input_thick_resized, vworld_thick_resized])
            sidebyside_thick_path = self.dirs['stage5'] / f"{self.image_name}_polyline_thick_sidebyside.png"
            cv2.imwrite(str(sidebyside_thick_path), sidebyside_thick)
            print(f"  ✓ Saved thick side-by-side: {sidebyside_thick_path}")
        
        # Convert smoothed pixels to GPS
        smoothed_gps = []
        for px, py in smoothed_pixels:
            lon, lat = self.vworld_pixel_to_gps(
                px, py,
                center_gps['lon'], center_gps['lat'],
                zoom, size
            )
            smoothed_gps.append([lon, lat])
        
        smoothed_gps = np.array(smoothed_gps)
        
        # Calculate statistics
        total_distance_m = 0
        for i in range(len(gps_polyline) - 1):
            dist = great_circle(
                (gps_polyline[i][1], gps_polyline[i][0]),
                (gps_polyline[i + 1][1], gps_polyline[i + 1][0])
            ).meters
            total_distance_m += dist
        
        gps_result = {
            "stage": "5",
            "description": "GPS coordinates of extracted highlight polyline",
            "timestamp": datetime.now().isoformat(),
            "polyline": {
                "coordinates": gps_polyline.tolist(),
                "num_points": len(gps_polyline),
                "format": "[[lon, lat], ...]",
                "coordinate_system": "WGS84 (EPSG:4326)"
            },
            "smoothed_polyline": {
                "coordinates": smoothed_gps.tolist(),
                "num_points": len(smoothed_gps),
                "smoothing_applied": True
            },
            "statistics": {
                "total_distance_meters": total_distance_m,
                "total_distance_km": total_distance_m / 1000,
                "num_segments": len(gps_polyline) - 1
            },
            "bounding_box": {
                "min_lon": float(gps_polyline[:, 0].min()),
                "max_lon": float(gps_polyline[:, 0].max()),
                "min_lat": float(gps_polyline[:, 1].min()),
                "max_lat": float(gps_polyline[:, 1].max())
            },
            "outputs": {
                "visualization_thin": str(vis_thin_path),
                "visualization_thick_smooth": str(vis_thick_path)
            }
        }
        
        # Add side-by-side paths if they exist
        if input_img is not None:
            gps_result["outputs"]["visualization_thin_sidebyside"] = str(sidebyside_thin_path)
            gps_result["outputs"]["visualization_thick_sidebyside"] = str(sidebyside_thick_path)
        
        # Save as JSON
        self.save_artifact(
            gps_result,
            f"{self.image_name}_stage5_gps_polyline.json",
            'stage5'
        )
        
        # Also save as GeoJSON for easy use with mapping tools
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": f"{self.image_name}_highlight",
                        "distance_km": gps_result['statistics']['total_distance_km']
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": gps_polyline.tolist()
                    }
                }
            ]
        }
        
        geojson_path = self.dirs['stage5'] / f"{self.image_name}_polyline.geojson"
        with open(geojson_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Saved GeoJSON: {geojson_path}")
        
        print(f"\n✓ Stage 5 Complete!")
        print(f"  GPS Polyline: {len(gps_polyline)} points")
        print(f"  Total distance: {gps_result['statistics']['total_distance_km']:.2f} km")
        print(f"  Bounding box: ({gps_result['bounding_box']['min_lon']:.6f}, {gps_result['bounding_box']['min_lat']:.6f}) to ({gps_result['bounding_box']['max_lon']:.6f}, {gps_result['bounding_box']['max_lat']:.6f})")
        
        return gps_result
    
    def _order_polyline_points(self, points: np.ndarray) -> np.ndarray:
        """
        Order points along polyline using nearest-neighbor
        
        Args:
            points: Unordered points [N, 2]
            
        Returns:
            Ordered indices
        """
        if len(points) < 2:
            return np.arange(len(points))
        
        # Start from leftmost point
        start_idx = np.argmin(points[:, 0])
        
        ordered_indices = [start_idx]
        remaining = set(range(len(points))) - {start_idx}
        
        current_idx = start_idx
        
        while remaining:
            current_point = points[current_idx]
            
            # Find nearest remaining point
            min_dist = float('inf')
            nearest_idx = None
            
            for idx in remaining:
                dist = np.linalg.norm(points[idx] - current_point)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
            
            ordered_indices.append(nearest_idx)
            remaining.remove(nearest_idx)
            current_idx = nearest_idx
        
        return np.array(ordered_indices)
    
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================


    def run(self, image_path: str, highlight_checkpoint: str = None) -> Dict:
        """Execute complete pipeline (Stages 1-5)"""
        print("\n" + "="*80)
        print(f"ROAD DETECTION PIPELINE: COMPLETE EXECUTION")
        print("="*80)
        print(f"Input: {image_path}")
        print(f"Output: {self.output_root}")
        print(f"Matching method: {self.matching_method}")
        if highlight_checkpoint:
            print(f"Highlight model: {highlight_checkpoint}")
        print("="*80 + "\n")
        
        self.step_counter = 0
        self.attempted_combinations.clear()
        
        try:
            # Copy original input image to output directory
            # (will be done after setup_directories is called in phase1_initial_ocr)
            
            # ====================================================================
            # STAGE 1: OCR → POI EXTRACTION → BOUNDING BOX
            # ====================================================================
            
            print("\n" + "="*80)
            print("STAGE 1: OCR → POI EXTRACTION → BOUNDING BOX")
            print("="*80 + "\n")
            
            # PHASE 1: Initial extraction + crop OCR + rule filter + LLM filter + ranking + DB search
            ocr_data = self.phase1_initial_ocr(image_path)
            
            # NOW copy the original image to the output root
            original_image_dest = self.dirs['root'] / f"{self.image_name}_original.jpg"
            shutil.copy2(image_path, original_image_dest)
            print(f"✓ Copied original input image to: {original_image_dest}")
            
            ocr_data = self.phase1_crop_ocr(ocr_data)
            ocr_data = self.phase1_rule_based_filtering(ocr_data)
            ocr_data = self.phase1_llm_filtering(ocr_data)
            ocr_data = self.phase1_llm_ranking(ocr_data)
            extraction_data = self.phase1_database_search(ocr_data)
            
            confirmed = extraction_data['summary']['found_in_db']
            
            print(f"\n{'='*80}")
            print(f"Phase 1 Complete: {confirmed} confirmed POIs")
            print('='*80)
            
            # Before trying bbox after Phase 1: filter by place.csv second column
            extraction_data = self.filter_place_csv_second_column(extraction_data, 'phase1')
            confirmed = extraction_data['summary']['found_in_db']

            # Try bbox after Phase 1
            bbox_result = None
            if confirmed >= 3:
                print(f"\n→ Trying bbox extraction...")
                bbox_result = self.try_bounding_box_extraction(extraction_data)
                if bbox_result:
                    # Stage 1 succeeded, proceed to Stage 2
                    print(f"\n{'='*80}")
                    print(f"STAGE 1 SUCCESS: Bounding box extracted at Phase 1")
                    print('='*80)
                    
                    stage1_result = {
                        "status": "success",
                        "phase": 1,
                        "extraction_data": extraction_data,
                        "bounding_box": bbox_result
                    }
                    
                    # Proceed to Stage 2 (choose method)
                    if self.matching_method == "affine":
                        transformation_result = self.stage2_affine_transformation(bbox_result)
                    elif self.matching_method == "feature":
                        transformation_result = self.stage2_feature_matching(bbox_result)
                    else:
                        raise ValueError(f"Unknown matching method: {self.matching_method}")
                    
                    result = {
                        "status": "success",
                        "stage1": stage1_result,
                        "stage2": transformation_result
                    }
                    
                    # Proceed to Stages 3-5 if highlight model provided
                    if highlight_checkpoint:
                        print("\n" + "="*80)
                        print("STAGES 3-5: HIGHLIGHT EXTRACTION & GPS CALCULATION")
                        print("="*80 + "\n")
                        
                        # Stage 3: Highlight extraction
                        try:
                            highlight_result = self.stage3_highlight_extraction(
                                image_path,
                                highlight_checkpoint
                            )
                            result['stage3'] = highlight_result
                        except Exception as e:
                            print(f"\n✗ Stage 3 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 3 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                        
                        # Stage 4: Map to VWorld
                        try:
                            mapping_result = self.stage4_map_highlight_to_vworld(
                                highlight_result,
                                result['stage2']
                            )
                            result['stage4'] = mapping_result
                        except Exception as e:
                            print(f"\n✗ Stage 4 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 4 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                        
                        # Stage 5: Calculate GPS
                        try:
                            gps_result = self.stage5_calculate_highlight_gps(
                                mapping_result,
                                result['stage2']
                            )
                            result['stage5'] = gps_result
                        except Exception as e:
                            print(f"\n✗ Stage 5 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 5 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                    
                    # Success - close log file
                    if hasattr(self, 'log_file_handle') and self.log_file_handle:
                        print(f"\n{'='*80}")
                        print(f"LOGGING ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                        print(f"{'='*80}\n")
                        sys.stdout = self.tee_logger.terminal
                        self.log_file_handle.close()
                    
                    return result
            
            # PHASE 2: All-in-one LLM revision
            extraction_data = self.phase2_all_in_one_revision(extraction_data)
            
            confirmed = extraction_data['summary']['found_in_db']
            
            print(f"\n{'='*80}")
            print(f"Phase 2 Complete: {confirmed} confirmed POIs")
            print('='*80)
            
            # Before trying bbox after Phase 2: filter by place.csv second column
            extraction_data = self.filter_place_csv_second_column(extraction_data, 'phase2')
            confirmed = extraction_data['summary']['found_in_db']

            # Try bbox after Phase 2
            if confirmed >= 3:
                print(f"\n→ Trying bbox extraction...")
                bbox_result = self.try_bounding_box_extraction(extraction_data)
                if bbox_result:
                    print(f"\n{'='*80}")
                    print(f"STAGE 1 SUCCESS: Bounding box extracted at Phase 2")
                    print('='*80)
                    
                    stage1_result = {
                        "status": "success",
                        "phase": 2,
                        "extraction_data": extraction_data,
                        "bounding_box": bbox_result
                    }
                    
                    # Proceed to Stage 2
                    if self.matching_method == "affine":
                        transformation_result = self.stage2_affine_transformation(bbox_result)
                    elif self.matching_method == "feature":
                        transformation_result = self.stage2_feature_matching(bbox_result)
                    else:
                        raise ValueError(f"Unknown matching method: {self.matching_method}")
                    
                    result = {
                        "status": "success",
                        "stage1": stage1_result,
                        "stage2": transformation_result
                    }
                    
                    # Proceed to Stages 3-5 if highlight model provided
                    if highlight_checkpoint:
                        print("\n" + "="*80)
                        print("STAGES 3-5: HIGHLIGHT EXTRACTION & GPS CALCULATION")
                        print("="*80 + "\n")
                        
                        # Stage 3: Highlight extraction
                        try:
                            highlight_result = self.stage3_highlight_extraction(
                                image_path,
                                highlight_checkpoint
                            )
                            result['stage3'] = highlight_result
                        except Exception as e:
                            print(f"\n✗ Stage 3 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 3 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                        
                        # Stage 4: Map to VWorld
                        try:
                            mapping_result = self.stage4_map_highlight_to_vworld(
                                highlight_result,
                                result['stage2']
                            )
                            result['stage4'] = mapping_result
                        except Exception as e:
                            print(f"\n✗ Stage 4 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 4 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                        
                        # Stage 5: Calculate GPS
                        try:
                            gps_result = self.stage5_calculate_highlight_gps(
                                mapping_result,
                                result['stage2']
                            )
                            result['stage5'] = gps_result
                        except Exception as e:
                            print(f"\n✗ Stage 5 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 5 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                    
                    # Success - close log file
                    if hasattr(self, 'log_file_handle') and self.log_file_handle:
                        print(f"\n{'='*80}")
                        print(f"LOGGING ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                        print(f"{'='*80}\n")
                        sys.stdout = self.tee_logger.terminal
                        self.log_file_handle.close()
                    
                    return result
            
            # PHASE 3: Individual crop-level LLM revision
            extraction_data = self.phase3_individual_revision(extraction_data)
            
            confirmed = extraction_data['summary']['found_in_db']
            
            print(f"\n{'='*80}")
            print(f"Phase 3 Complete: {confirmed} confirmed POIs")
            print('='*80)
            
            # Before final bbox attempt: filter by place.csv second column
            extraction_data = self.filter_place_csv_second_column(extraction_data, 'phase3_final')
            confirmed = extraction_data['summary']['found_in_db']

            # Final bbox attempt
            if confirmed >= 3:
                print(f"\n→ Final bbox extraction attempt...")
                bbox_result = self.try_bounding_box_extraction(extraction_data)
                if bbox_result:
                    print(f"\n{'='*80}")
                    print(f"STAGE 1 SUCCESS: Bounding box extracted at Phase 3")
                    print('='*80)
                    
                    stage1_result = {
                        "status": "success",
                        "phase": 3,
                        "extraction_data": extraction_data,
                        "bounding_box": bbox_result
                    }
                    
                    # Proceed to Stage 2
                    if self.matching_method == "affine":
                        transformation_result = self.stage2_affine_transformation(bbox_result)
                    elif self.matching_method == "feature":
                        transformation_result = self.stage2_feature_matching(bbox_result)
                    else:
                        raise ValueError(f"Unknown matching method: {self.matching_method}")
                    
                    result = {
                        "status": "success",
                        "stage1": stage1_result,
                        "stage2": transformation_result
                    }
                    
                    # Proceed to Stages 3-5 if highlight model provided
                    if highlight_checkpoint:
                        print("\n" + "="*80)
                        print("STAGES 3-5: HIGHLIGHT EXTRACTION & GPS CALCULATION")
                        print("="*80 + "\n")
                        
                        # Stage 3: Highlight extraction
                        try:
                            highlight_result = self.stage3_highlight_extraction(
                                image_path,
                                highlight_checkpoint
                            )
                            result['stage3'] = highlight_result
                        except Exception as e:
                            print(f"\n✗ Stage 3 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 3 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                        
                        # Stage 4: Map to VWorld
                        try:
                            mapping_result = self.stage4_map_highlight_to_vworld(
                                highlight_result,
                                result['stage2']
                            )
                            result['stage4'] = mapping_result
                        except Exception as e:
                            print(f"\n✗ Stage 4 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 4 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                        
                        # Stage 5: Calculate GPS
                        try:
                            gps_result = self.stage5_calculate_highlight_gps(
                                mapping_result,
                                result['stage2']
                            )
                            result['stage5'] = gps_result
                        except Exception as e:
                            print(f"\n✗ Stage 5 failed: {e}")
                            import traceback
                            traceback.print_exc()
                            
                            # Close log file before returning
                            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                                print(f"\n{'='*80}")
                                print(f"LOGGING ENDED (with Stage 5 error): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                                print(f"{'='*80}\n")
                                sys.stdout = self.tee_logger.terminal
                                self.log_file_handle.close()
                            
                            return result
                    
                    # Success - close log file
                    if hasattr(self, 'log_file_handle') and self.log_file_handle:
                        print(f"\n{'='*80}")
                        print(f"LOGGING ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                        print(f"{'='*80}\n")
                        sys.stdout = self.tee_logger.terminal
                        self.log_file_handle.close()
                    
                    return result
            
            print(f"\n✗ Pipeline complete but insufficient data: {confirmed} POIs (need 3+)")
            
            result = {
                "status": "insufficient_data",
                "phase": "final",
                "extraction_data": extraction_data,
                "message": f"Only {confirmed} confirmed POIs after all phases"
            }
            
            # Close log file
            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                print(f"\n{'='*80}")
                print(f"LOGGING ENDED (insufficient data): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Complete log saved to: {self.dirs['logs']}/{self.image_name}_complete_output.log")
                print(f"{'='*80}\n")
                sys.stdout = self.tee_logger.terminal
                self.log_file_handle.close()
            
            return result
            
        except Exception as e:
            # Make sure to close log file on error
            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                print(f"\n{'='*80}")
                print(f"ERROR OCCURRED - LOGGING ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Error: {e}")
                print(f"{'='*80}\n")
                import traceback
                traceback.print_exc()
                sys.stdout = self.tee_logger.terminal
                self.log_file_handle.close()
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Road Detection Pipeline: Complete End-to-End System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  Stage 1: OCR → POI Extraction → Bounding Box from VWorld
    - Phase 1: OCR + Filtering + Ranking + Database Search
    - Phase 2: All-in-One LLM Revision
    - Phase 3: Individual Crop-level LLM Revision
    - Phase 4: Bounding Box Extraction & Validation
  
  Stage 2: Map Alignment (affine transformation or feature matching)
  Stage 3: Highlight Extraction from input map (requires --highlight-model)
  Stage 4: Map Highlight to VWorld Bounding Box
  Stage 5: Calculate GPS Coordinates of Highlight Polyline

Example Usage:
  # Stages 1-2 only (no highlight extraction)
  python road_detection_pipeline.py input.jpg -d database.parquet --llm qwen
  
  # Complete pipeline (Stages 1-5)
  python road_detection_pipeline.py input.jpg -d database.parquet --llm qwen \\
    --highlight-model /path/to/road_unet.pth

Feature Matching Methods:
  affine:  POI-based affine transformation (requires 3+ POIs)
  feature: Advanced sliding window matching using:
           - Road mask, Water mask
           - Canny edges, Sobel gradients
           - Orientation maps
           - ZNCC, Chamfer distance, Orientation metrics
"""
    )
    
    parser.add_argument('image', help='Input map image')
    parser.add_argument('-o', '--output', default='./road_detection_result',
                       help='Output directory (default: ./road_detection_result)')
    parser.add_argument('-d', '--database', required=True,
                       help='Korean Place Name Database (parquet)')
    parser.add_argument('-k', '--api-key', default=None,
                       help='VWorld API key (optional, will use default if not provided)')
    parser.add_argument('--llm', choices=['qwen', 'gpt'], default='qwen',
                       help='LLM type (default: qwen)')
    parser.add_argument('--gpt-api-key', default=None,
                       help='OpenAI API key (required for GPT mode)')
    parser.add_argument('--reasoning', choices=['minimal', 'low', 'medium', 'high'], 
                       default='minimal',
                       help='GPT reasoning capacity (default: minimal, only for GPT mode)')
    parser.add_argument('--matching-method', choices=['affine', 'feature'], 
                       default='affine',
                       help='Stage 2 matching method (default: affine)')
    parser.add_argument('--highlight-model', default=None,
                       help='Path to highlight extraction model checkpoint (.pth) for Stages 3-5')
    
    args = parser.parse_args()
    
    # Validation
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return 1
    
    if not Path(args.database).exists():
        print(f"Error: Database not found: {args.database}")
        return 1
    
    if args.highlight_model and not Path(args.highlight_model).exists():
        print(f"Error: Highlight model not found: {args.highlight_model}")
        return 1
    
    if args.llm == 'gpt' and not args.gpt_api_key:
        if not os.environ.get('OPENAI_API_KEY'):
            print(f"Error: GPT mode requires --gpt-api-key or OPENAI_API_KEY environment variable")
            return 1
    
    try:
        pipeline = RoadDetectionPipeline(
            output_dir=args.output,
            database_path=args.database,
            vworld_api_key=args.api_key,
            llm_type=args.llm,
            gpt_api_key=args.gpt_api_key,
            reasoning_effort=args.reasoning,
            matching_method=args.matching_method
        )
        
        result = pipeline.run(args.image, highlight_checkpoint=args.highlight_model)
        
        # Prepare result for JSON serialization (remove numpy arrays)
        result_to_save = result.copy()
        if 'stage3' in result_to_save and 'mask' in result_to_save['stage3']:
            # Remove mask (numpy array) before saving
            result_to_save['stage3'] = result_to_save['stage3'].copy()
            del result_to_save['stage3']['mask']
        
        # Save final result
        final_path = pipeline.dirs['root'] / f"{pipeline.image_name}_FINAL_RESULT.json"
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(result_to_save, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print("ROAD DETECTION PIPELINE COMPLETE")
        print("="*80)
        print(f"Status: {result['status']}")
        print(f"Final result: {final_path}")
        
        if result['status'] == 'success':
            stage1 = result['stage1']
            stage2 = result['stage2']
            
            print(f"\n✓ STAGE 1 Success!")
            print(f"  Phase: {stage1['phase']}")
            print(f"  Strategy: {stage1['bounding_box']['strategy']}")
            print(f"  Confirmed POIs: {len(stage1['bounding_box']['confirmed_pois'])}")
            print(f"  VWorld image: {stage1['bounding_box']['vworld_image_path']}")
            print(f"  Marked VWorld: {stage1['bounding_box']['vworld_marked_image_path']}")
            
            method = stage2['method']
            print(f"\n✓ STAGE 2 Success ({method})!")
            
            if method == 'affine_transformation':
                print(f"  Method: Affine Transformation (POI-based)")
                print(f"  Overlapped image: {stage2['outputs']['overlapped_image']}")
                
                print(f"\n  POI Correspondences:")
                for pt in stage2['correspondence_points']:
                    print(f"    {pt['poi_name']}:")
                    print(f"      GPS:    ({pt['gps']['lon']:.6f}, {pt['gps']['lat']:.6f})")
                    
            elif method == 'feature_matching':
                print(f"  Method: Advanced Feature Matching")
                print(f"  Features: Road, Water, Edges, Gradients, Orientation")
                print(f"  Metrics: ZNCC, Chamfer, Orientation")
                print(f"  Overlapped image: {stage2['outputs']['overlapped_image']}")
                
                best_match = stage2['best_match']
                print(f"\n  Best Match:")
                print(f"    Position: ({best_match['position']['x']}, {best_match['position']['y']})")
                print(f"    Composite Score: {best_match['composite_score']:.4f}")
                print(f"\n  Individual Scores:")
                for metric, score in best_match['individual_scores'].items():
                    print(f"    {metric}: {score:.4f}")
            
            print(f"\n  Stage 2 outputs: {pipeline.dirs['stage2']}")
            
            # Stage 3-5 reporting
            if result.get('stage3'):
                stage3 = result['stage3']
                print(f"\n✓ STAGE 3 Success!")
                print(f"  Method: Highlight Extraction (DeepLabV3)")
                print(f"  Skeleton pixels: {stage3['num_skeleton_pixels']}")
                print(f"  Outputs:")
                print(f"    - Raw mask: {stage3['outputs']['raw_mask']}")
                print(f"    - Skeleton: {stage3['outputs']['skeleton']}")
                print(f"    - Overlay: {stage3['outputs']['skeleton_overlay']}")
            
            if result.get('stage4'):
                stage4 = result['stage4']
                print(f"\n✓ STAGE 4 Success!")
                print(f"  Method: Highlight Mapping to VWorld")
                print(f"  Mapped pixels: {stage4['vworld_mapped_pixels']} / {stage4['input_skeleton_pixels']}")
                print(f"  Outputs:")
                print(f"    - Initial mask on input: {stage4['outputs']['initial_highlight_on_input']}")
                print(f"    - Initial mask on VWorld: {stage4['outputs']['initial_highlight_on_vworld']}")
                print(f"    - Side-by-side: {stage4['outputs']['initial_sidebyside']}")
                print(f"    - Final highlight: {stage4['outputs']['highlight_on_vworld']}")
            
            if result.get('stage5'):
                stage5 = result['stage5']
                print(f"\n✓ STAGE 5 Success!")
                print(f"  Method: GPS Coordinate Calculation")
                print(f"  GPS Polyline: {stage5['polyline']['num_points']} points")
                print(f"  Total distance: {stage5['statistics']['total_distance_km']:.2f} km")
                print(f"  Bounding box:")
                bbox = stage5['bounding_box']
                print(f"    Lon: {bbox['min_lon']:.6f} to {bbox['max_lon']:.6f}")
                print(f"    Lat: {bbox['min_lat']:.6f} to {bbox['max_lat']:.6f}")
                print(f"  Outputs:")
                print(f"    - GeoJSON: {pipeline.dirs['stage5']}/{pipeline.image_name}_polyline.geojson")
                print(f"    - Thin polyline: {stage5['outputs']['visualization_thin']}")
                print(f"    - Thick smooth: {stage5['outputs']['visualization_thick_smooth']}")
                if 'visualization_thin_sidebyside' in stage5['outputs']:
                    print(f"    - Side-by-side (thin): {stage5['outputs']['visualization_thin_sidebyside']}")
                    print(f"    - Side-by-side (thick): {stage5['outputs']['visualization_thick_sidebyside']}")
                
                print(f"\n  🎉 Complete pipeline finished successfully!")
                print(f"  📍 Final GPS polyline saved to GeoJSON format")
                print(f"  📊 All outputs available in: {pipeline.dirs['root']}")
        else:
            print(f"\n✗ {result['message']}")
            print(f"\n  Partial results saved to: {pipeline.dirs['root']}")
        
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
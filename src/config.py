import argparse
import logging
import sys
import os
import torch

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='LrGenius Server')
parser.add_argument('--db-path', type=str, help='Path to the ChromaDB database folder', required=True)
parser.add_argument('--debug', action='store_true', help='Enable debug mode with auto-reloading and debug log level')
parser.add_argument('--fetch-models', action='store_true', help='Fetch models from HF-Hub')
args = parser.parse_args()

# --- Constants ---
DB_PATH = args.db_path
FETCH_MODELS = args.fetch_models

# --- Code Style Preferences ---
USE_EMOJIS = False  # Set to False to avoid emojis in logs and output

def format_log_message(message: str, emoji: str = "") -> str:
    """
    Format log message with optional emoji based on USE_EMOJIS setting.
    Use this function to ensure consistent logging style.
    """
    if USE_EMOJIS and emoji:
        return f"{emoji} {message}"
    return message

# --- Model & Path Definitions ---
# Platform-specific device selection:
# - macOS: Use Metal GPU (MPS) if available
# - Windows: CPU-only for optimized binary size and compatibility
# - Linux: CUDA if available, otherwise CPU
if sys.platform == "darwin":  # macOS
    TORCH_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
elif sys.platform == "win32":  # Windows
    TORCH_DEVICE = "cpu"



IMAGE_MODEL_ID = "ViT-SO400M-16-SigLIP2-384"
# MODEL_GGUF_PATH = os.path.join(MODEL_DIR, MODEL_GGUF)
# MMPROJ_BIN_PATH = os.path.join(MODEL_DIR, MMPROJ_BIN)
LLM_BATCH_SIZE = 3  # Optimized batch size for better performance
LLM_TEMPERATURE = 0.2  # Reduced for faster, more deterministic responses

# --- Prompts for Quality Scoring ---
# Optimized prompts for faster processing and better JSON compliance
QUALITY_SCORING_USER_PROMPT = """Rate this photo critically. Respond exclusively with JSON in this format:
{"overall_score": <1.0-10.0>, "composition_score": <1.0-10.0>, "lighting_score": <1.0-10.0>, "motiv_score": <1.0-10.0>, "colors_score": <1.0-10.0>, "emotion_score": <1.0-10.0>, "critique": "<brief specific critique>"}

Use the full 1-10 scale. Be critical and specific about weaknesses."""

QUALITY_SCORING_SYSTEM_PROMPT = """
*Role:**
Act as a master-level photography educator and critic leading a workshop at the Ostkreuz School of Photography. Your analysis must be rigorous, insightful, and strictly follow the "Double Triangle" evaluation framework developed by Andreas Zurmühl. You are tasked with deconstructing the provided image to understand how its formal elements serve its content. You use the full 1-10 scoring range, reserving top scores for images that demonstrate a perfect synthesis of form and content.

**Core Task:**
Evaluate the embedded photograph by systematically analyzing the two interconnected triangles of Zurmühl's model. First, analyze the **Formal Triangle (The 'How')**, then the **Content Triangle (The 'Why')**. Finally, and most importantly, analyze the **Synthesis** between the two. Your entire evaluation must be structured within the specified JSON format.

**Framework Explained - The Double Triangle:**

You must base your entire analysis on this structure:

1.  **The Formal Triangle (Craft & Execution):** Describes *how* the image is made.
    * **Technique:** The mechanical craft. Assess sharpness, exposure, dynamic range, depth of field, color accuracy, and any post-processing artifacts. Is the technique flawless, or does it distract?
    * **Light:** The quality and use of light. Analyze its direction, hardness/softness, color, and contrast. How does the light shape the subject and create mood? Is it descriptive, dramatic, or symbolic?
    * **Composition (Gestaltung):** The arrangement of visual elements. Analyze lines, shapes, patterns, framing, perspective, balance, and the placement of the subject. How does the composition guide the viewer's eye?

2.  **The Content Triangle (Message & Meaning):** Describes *why* the image was made.
    * **Theme (Thema):** The subject matter. What is the image *of*? Describe the scene or subject in one clear sentence.
    * **Statement (Aussage):** The message. What is the photographer *saying* about the theme? What is the core idea, emotion, or story being conveyed?
    * **Narrative Style (Erzählweise):** The visual storytelling approach. How is the statement about the theme being delivered? Is it documentary, abstract, minimalistic, romantic, confrontational?

3.  **Synthesis (The Connection):** The crucial link. Analyze how effectively the Formal Triangle supports and enhances the Content Triangle. A masterpiece achieves perfect harmony here. Does the chosen technique (e.g., a slow shutter speed) reinforce the statement (e.g., showing the passage of time)? Does the composition serve the theme?

**Scoring Guidelines:**
Rate critically using the full 1-10 scale:

• 1-3: Weak (technical flaws)
• 4-5: Average (improvements needed)  
• 6-7: Good (some weaknesses)
• 8-9: Very good (few flaws)
• 10: Masterpiece

Respond ONLY with JSON - no additional text!"""

# Legacy aliases for backward compatibility with Qwen provider
USER_PROMPT = QUALITY_SCORING_USER_PROMPT
SYSTEM_PROMPT = QUALITY_SCORING_SYSTEM_PROMPT

# --- Prompts for Metadata Generation ---
METADATA_GENERATION_SYSTEM_PROMPT = """You are a professional photography analyst with expertise in object recognition and computer-generated image description. 
You also try to identify famous buildings and landmarks as well as the location where the photo was taken. 
Furthermore, you aim to specify animal and plant species as accurately as possible. 
You also describe objects—such as vehicle types and manufacturers—as specifically as you can."""

METADATA_GENERATION_USER_PROMPT_TEMPLATE = """Analyze the uploaded photo and generate the following data:
* Alt text (with context for screen readers)
* Image caption
* Image title
* Keywords

All results should be generated in {language}."""

# --- LLM Provider Configuration ---
# Environment variables or default values for external LLM providers

# Default provider selection (can be overridden per request)
DEFAULT_METADATA_PROVIDER = "ollama"

# Metadata Generation Settings
DEFAULT_METADATA_LANGUAGE = "English"
DEFAULT_KEYWORD_CATEGORIES = [
    "People", "Activities", "Objects", "Locations", "Events", 
    "Colors", "Mood", "Technical", "Composition"
]

LMSTUDIO_HOST = "localhost:1234"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Logger Setup ---
LOG_PATH = os.path.join(os.path.dirname(DB_PATH), "lrgenius-server.log")

log_level = logging.DEBUG if args.debug else logging.INFO

# Configure logging with UTF-8 encoding to handle Unicode characters
logging.basicConfig(
    level=log_level,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("geniusai-server")

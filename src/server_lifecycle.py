import os
import time
import signal
from config import DB_PATH, logger, IMAGE_MODEL_ID, TORCH_DEVICE
import sys
import open_clip
import threading
import datetime
import gc
import torch

# Lazy-loadable global model instances
# model, processor and tokenizer start as None and will be loaded on first use.
model = None
processor = None  # This will hold the image preprocessor
tokenizer = None

# Idle-unload handling
# If the model hasn't been used for this many seconds, it will be unloaded to free memory.
IDLE_UNLOAD_SECONDS = 30 * 60  # 30 minutes
_last_used = None
_model_lock = threading.RLock()
_unloader_thread = None


def _set_last_used():
    global _last_used
    _last_used = datetime.datetime.utcnow()


def _needs_unload():
    if _last_used is None:
        return False
    delta = datetime.datetime.utcnow() - _last_used
    return delta.total_seconds() >= IDLE_UNLOAD_SECONDS


def load_model():
    """Load the OpenCLIP model (idempotent)."""
    global model, processor, tokenizer
    with _model_lock:
        if model is not None:
            _set_last_used()
            return


        try:
            # Determine the path to the bundled model directory
            # For PyInstaller, check if running as bundled executable
            if getattr(sys, 'frozen', False):
                # Running as PyInstaller bundle - use executable directory
                server_dir = os.path.dirname(sys.executable)
            else:
                # Running as script - use script directory
                server_dir = os.path.dirname(os.path.abspath(__file__))
                # Go up one level from src/ to server root
                server_dir = os.path.dirname(server_dir)
            
            local_model_dir = os.path.join(os.path.dirname(server_dir), 'models')
            
            logger.info(f"Server directory: {server_dir}")
            logger.info(f"Checking for bundled model at: {local_model_dir}")
            
            # Check if local model directory exists (production/bundled scenario)
            if os.path.isdir(local_model_dir):
                # Verify model files exist
                config_file = os.path.join(local_model_dir, 'open_clip_config.json')
                weights_file = os.path.join(local_model_dir, 'open_clip_model.safetensors')
                
                if os.path.isfile(config_file) and os.path.isfile(weights_file):
                    local_model_uri = f"local-dir:{local_model_dir}"
                    logger.info(f"Loading OpenCLIP model from bundled directory: {local_model_dir}")
                    model_obj, _, proc = open_clip.create_model_and_transforms(
                        local_model_uri,
                        pretrained=None
                    )
                    tok = open_clip.get_tokenizer(local_model_uri)
                else:
                    logger.warning(f"Bundled model directory exists but required files missing")
                    logger.warning(f"Config file exists: {os.path.isfile(config_file)}")
                    logger.warning(f"Weights file exists: {os.path.isfile(weights_file)}")
                    raise FileNotFoundError("Bundled model files incomplete")
            else:
                # Fallback to HuggingFace Hub for development
                logger.info(f"Bundled model directory not found at {local_model_dir}")
                logger.info(f"Falling back to HuggingFace Hub (development mode)")
                logger.info(f"Loading OpenCLIP model '{IMAGE_MODEL_ID}' from HuggingFace...")
                model_obj, _, proc = open_clip.create_model_and_transforms(
                    IMAGE_MODEL_ID,
                    pretrained='webli'
                )
                tok = open_clip.get_tokenizer(IMAGE_MODEL_ID)

            try:
                model_obj.to(TORCH_DEVICE)
                logger.info(f"Text and vision model moved to {TORCH_DEVICE}")
            except Exception as e:
                logger.warning(f"Failed to move text and vision model to {TORCH_DEVICE}: {e}.")

            model = model_obj
            processor = proc
            tokenizer = tok

            _set_last_used()
            logger.info("Loaded OpenCLIP model (lazy)")
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model (lazy): {e}", exc_info=True)
            raise


def unload_model():
    """Unload the loaded model to free GPU/CPU memory."""
    global model, processor, tokenizer
    with _model_lock:
        if model is None and processor is None and tokenizer is None:
            return

        logger.info("Unloading OpenCLIP model due to inactivity...")
        try:
            # If the model is a torch module, try moving it to cpu and delete the reference.
            try:
                if hasattr(model, "to"):
                    model.to("cpu")
            except Exception:
                pass

            model = None
            processor = None
            tokenizer = None

            # Best-effort free memory for CUDA
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            # Force a GC pass
            gc.collect()
            logger.info("Unloaded OpenCLIP model.")
        except Exception as e:
            logger.warning(f"Error while unloading model: {e}")


def _idle_unloader_loop():
    """Background thread which periodically checks whether the model should be unloaded."""
    global _unloader_thread
    logger.info("Starting server_lifecycle idle unloader thread")
    try:
        while True:
            time.sleep(60)
            try:
                if _needs_unload():
                    unload_model()
            except Exception:
                logger.exception("Error checking/unloading model in background thread")
    finally:
        logger.info("Server_lifecycle idle unloader thread exiting")


def _ensure_unloader_thread():
    global _unloader_thread
    if _unloader_thread is None or not _unloader_thread.is_alive():
        _unloader_thread = threading.Thread(target=_idle_unloader_loop, daemon=True, name="server_lifecycle_unloader")
        _unloader_thread.start()


def get_model():
    """Return the model, loading it lazily if needed."""
    load_model()
    _ensure_unloader_thread()
    return model


def get_processor():
    load_model()
    _ensure_unloader_thread()
    return processor


def get_tokenizer():
    load_model()
    _ensure_unloader_thread()
    return tokenizer

def get_db_dir():
    return os.path.dirname(DB_PATH)

def write_pid_file():
    pid_file = os.path.join(get_db_dir(), "lrgenius-server.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

def remove_pid_file():
    pid_file = os.path.join(get_db_dir(), "lrgenius-server.pid")
    try:
        os.remove(pid_file)
    except FileNotFoundError:
        pass

def write_ok_file():
    ok_file = os.path.join(get_db_dir(), "lrgenius-server.OK")
    with open(ok_file, "w") as f:
        f.write("OK\n")

def remove_ok_file():
    ok_file = os.path.join(get_db_dir(), "lrgenius-server.OK")
    try:
        os.remove(ok_file)
    except FileNotFoundError:
        pass

def request_shutdown():
    logger.info("Shutdown request received")
    time.sleep(1) # Give time for the response to be sent
    os.kill(os.getpid(), signal.SIGINT)

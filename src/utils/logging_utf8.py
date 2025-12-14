"""Utility để setup logging với UTF-8 encoding cho tiếng Việt."""
import sys
import io
import os
import logging
from typing import Optional


def fix_console_encoding():
    """Fix console encoding cho Windows để hiển thị tiếng Việt."""
    if sys.platform == 'win32':
        try:
            # Set console code page to UTF-8
            os.system('chcp 65001 >nul 2>&1')
        except:
            pass
        
        # Reconfigure stdout/stderr for UTF-8
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )


class UTF8StreamHandler(logging.StreamHandler):
    """StreamHandler với UTF-8 encoding support cho tiếng Việt."""
    
    def emit(self, record):
        """Emit log record với UTF-8 encoding."""
        try:
            msg = self.format(record)
            stream = self.stream
            
            # Ensure UTF-8 encoding
            if isinstance(msg, str):
                try:
                    # Try to write as-is (stdout/stderr đã được fix UTF-8)
                    stream.write(msg + self.terminator)
                except UnicodeEncodeError:
                    # Fallback: encode to UTF-8 bytes then decode
                    try:
                        msg_bytes = msg.encode('utf-8', errors='replace')
                        stream.write(msg_bytes.decode('utf-8', errors='replace') + self.terminator)
                    except:
                        # Last resort: use repr
                        stream.write(repr(msg) + self.terminator)
            else:
                stream.write(str(msg) + self.terminator)
            
            self.flush()
        except Exception:
            self.handleError(record)


def setup_utf8_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    file_encoding: str = 'utf-8'
):
    """
    Setup logging với UTF-8 encoding cho tiếng Việt.
    
    Args:
        level: Logging level (default: logging.INFO)
        format_string: Log format string (default: standard format)
        log_file: Optional log file path (default: None, only console)
        file_encoding: File encoding (default: 'utf-8')
    """
    # Fix console encoding first
    fix_console_encoding()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    handlers = []
    
    # Console handler với UTF-8
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler nếu có
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding=file_encoding, errors='replace')
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True  # Override existing config
    )
    
    return root_logger


def get_utf8_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get logger với UTF-8 support.
    
    Args:
        name: Logger name
        level: Logging level (default: logging.INFO)
    
    Returns:
        Logger instance
    """
    # Ensure console encoding is fixed
    fix_console_encoding()
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Ensure handlers use UTF-8 stdout
    if not logger.handlers:
        handler = UTF8StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    else:
        # Update existing handlers to use UTF-8 stdout
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.__stdout__:
                handler.stream = sys.stdout
    
    return logger














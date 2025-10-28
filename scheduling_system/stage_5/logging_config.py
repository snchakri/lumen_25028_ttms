#!/usr/bin/env python3
"""
Logging Configuration for Stage 5

Implements dual-output logging:
1. Console: Human-readable with colored output
2. File: JSON structured logs

Author: LUMEN TTMS
Version: 2.0.0
"""

import structlog
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def configure_stage5_logging(
    log_dir: Path,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_to_console: bool = True,
    log_to_file: bool = True
) -> None:
    """
    Configure dual-output logging for Stage 5.
    
    Args:
        log_dir: Directory for log files
        console_level: Logging level for console (INFO, DEBUG, WARNING, ERROR)
        file_level: Logging level for file (typically DEBUG for full detail)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Console output (human-readable)
    if log_to_console:
        console_processors = processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
        
        structlog.configure(
            processors=console_processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Set console log level
        import logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, console_level.upper())
        )
    
    # File output (JSON structured)
    if log_to_file:
        log_file = log_dir / f"stage5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        file_processors = processors + [
            structlog.processors.JSONRenderer()
        ]
        
        # Configure file logging
        import logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, file_level.upper()))
        
        logging.basicConfig(
            level=getattr(logging, file_level.upper()),
            handlers=[file_handler],
            format="%(message)s"
        )
        
        # Configure structlog for file output
        structlog.configure(
            processors=file_processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Log configuration
    logger = structlog.get_logger("stage5_logging")
    logger.info("Stage 5 logging configured",
               console_level=console_level,
               file_level=file_level,
               log_to_console=log_to_console,
               log_to_file=log_to_file,
               log_file=str(log_file) if log_to_file else None)

def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured logger for a component.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)



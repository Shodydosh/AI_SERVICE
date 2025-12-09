"""Vietnamese to English translator with protection for proper nouns, abbreviations, and technical terms."""
import re
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Try to import different translation backends
BACKEND_AVAILABLE = {
    'argostranslate': False,
    'easynmt': False,
    'transformers': False,
    'deep_translator': False
}

# Try argostranslate (offline, open source, recommended)
try:
    import argostranslate.package
    import argostranslate.translate
    BACKEND_AVAILABLE['argostranslate'] = True
except ImportError:
    pass

# Try EasyNMT (neural models, can run offline)
try:
    from easynmt import EasyNMT
    BACKEND_AVAILABLE['easynmt'] = True
except ImportError:
    pass

# Try transformers with Helsinki-NLP models
try:
    from transformers import MarianMTModel, MarianTokenizer
    BACKEND_AVAILABLE['transformers'] = True
except ImportError:
    pass

# Try deep_translator (Google Translate, requires internet)
try:
    from deep_translator import GoogleTranslator
    BACKEND_AVAILABLE['deep_translator'] = True
except ImportError:
    pass


class VietnameseTranslator:
    """Translator that preserves proper nouns, company names, certificates, products, and abbreviations."""
    
    # Common Vietnamese patterns that should be preserved
    PRESERVE_PATTERNS = [
        # Abbreviations (uppercase letters, numbers, dots)
        r'\b[A-Z]{2,}(?:\.[A-Z]+)*\b',  # AWS, React, CCNA, CCNP, MCSE, etc.
        # Company names (common Vietnamese companies)
        r'\b(?:Viettel|VinGroup|FPT|VNG|MobiFone|Vietnam|VNPT|Vietcombank|BIDV|Techcombank)\b',
        # Certificates
        r'\b(?:CCNA|CCNP|MCSE|AWS|Azure|GCP|PMP|CISSP|TOEIC|IELTS|TOEFL)\b',
        # Product names
        r'\b(?:React|Vue|Angular|Node\.?js|Python|Java|JavaScript|TypeScript|Go|Rust|PHP|Ruby)\b',
        # Common technical terms that should stay
        r'\b(?:API|REST|GraphQL|gRPC|SOAP|JSON|XML|HTML|CSS|SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Redis)\b',
    ]
    
    def __init__(self, backend: str = 'auto'):
        """
        Initialize translator.
        
        Args:
            backend: Translation backend to use. Options:
                - 'auto': Automatically choose best available backend
                - 'argostranslate': Offline, open source (recommended)
                - 'easynmt': Neural models, can run offline
                - 'transformers': Helsinki-NLP models
                - 'deep_translator': Google Translate (requires internet)
        """
        self.backend_type = None
        self.translator = None
        self._init_backend(backend)
    
    def _init_backend(self, backend: str):
        """Initialize translation backend."""
        if backend == 'auto':
            # Priority order: argostranslate > easynmt > transformers > deep_translator
            if BACKEND_AVAILABLE['argostranslate']:
                backend = 'argostranslate'
            elif BACKEND_AVAILABLE['easynmt']:
                backend = 'easynmt'
            elif BACKEND_AVAILABLE['transformers']:
                backend = 'transformers'
            elif BACKEND_AVAILABLE['deep_translator']:
                backend = 'deep_translator'
            else:
                logger.warning("No translation backend available. Install one of: argostranslate, easynmt, transformers, deep-translator")
                return
        
        if backend == 'argostranslate' and BACKEND_AVAILABLE['argostranslate']:
            try:
                self._init_argostranslate()
                self.backend_type = 'argostranslate'
                logger.info("Using argostranslate backend (offline)")
            except Exception as e:
                logger.warning(f"Failed to initialize argostranslate: {e}")
        
        elif backend == 'easynmt' and BACKEND_AVAILABLE['easynmt']:
            try:
                self._init_easynmt()
                self.backend_type = 'easynmt'
                logger.info("Using EasyNMT backend")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyNMT: {e}")
        
        elif backend == 'transformers' and BACKEND_AVAILABLE['transformers']:
            try:
                self._init_transformers()
                self.backend_type = 'transformers'
                logger.info("Using transformers (Helsinki-NLP) backend")
            except Exception as e:
                logger.warning(f"Failed to initialize transformers: {e}")
        
        elif backend == 'deep_translator' and BACKEND_AVAILABLE['deep_translator']:
            try:
                self.translator = GoogleTranslator(source='vi', target='en')
                self.backend_type = 'deep_translator'
                logger.info("Using deep_translator (Google Translate) backend")
            except Exception as e:
                logger.warning(f"Failed to initialize deep_translator: {e}")
        
        if not self.translator:
            logger.warning("No translation backend initialized. Translation will be skipped.")
    
    def _init_argostranslate(self):
        """Initialize argostranslate backend."""
        import argostranslate.package
        import argostranslate.translate
        
        # Download and install language package if needed
        installed_languages = argostranslate.package.get_installed_languages()
        from_code = 'vi'
        to_code = 'en'
        
        # Check if Vietnamese to English package is installed
        package_installed = False
        for lang in installed_languages:
            if lang.code == from_code:
                package_installed = True
                break
        
        if not package_installed:
            logger.info("Downloading Vietnamese to English translation package (first time only)...")
            try:
                argostranslate.package.update_package_index()
                available_packages = argostranslate.package.get_available_packages()
                package_to_install = next(
                    (pkg for pkg in available_packages 
                     if pkg.from_code == from_code and pkg.to_code == to_code),
                    None
                )
                if package_to_install:
                    argostranslate.package.install_from_path(package_to_install.download())
                    logger.info("Translation package installed successfully")
                else:
                    logger.warning(f"Vietnamese to English package not found. Using English as fallback.")
            except Exception as e:
                logger.warning(f"Failed to install translation package: {e}. Translation may not work.")
        
        self.translator = 'argostranslate'  # Mark as available
    
    def _init_easynmt(self):
        """Initialize EasyNMT backend."""
        # Use a lightweight model for Vietnamese-English
        self.translator = EasyNMT('opus-mt')  # or 'm2m_100_418M' for better quality
    
    def _init_transformers(self):
        """Initialize transformers backend with Helsinki-NLP model."""
        model_name = 'Helsinki-NLP/opus-mt-vi-en'
        self.translator = {
            'model': MarianMTModel.from_pretrained(model_name),
            'tokenizer': MarianTokenizer.from_pretrained(model_name)
        }
    
    def _detect_vietnamese(self, text: str) -> bool:
        """Detect if text contains Vietnamese characters."""
        if not text:
            return False
        # Vietnamese Unicode range
        vietnamese_pattern = re.compile(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]')
        return bool(vietnamese_pattern.search(text))
    
    def _extract_preserved_items(self, text: str) -> Dict[str, str]:
        """Extract items to preserve and replace with placeholders."""
        preserved = {}
        placeholder_map = {}
        counter = 0
        
        # Combine all patterns
        all_patterns = '|'.join(self.PRESERVE_PATTERNS)
        
        def replace_with_placeholder(match):
            nonlocal counter
            original = match.group(0)
            placeholder = f"__PRESERVE_{counter}__"
            placeholder_map[placeholder] = original
            counter += 1
            return placeholder
        
        # Replace preserved items with placeholders
        processed_text = re.sub(all_patterns, replace_with_placeholder, text, flags=re.IGNORECASE)
        
        return processed_text, placeholder_map
    
    def _restore_preserved_items(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore preserved items from placeholders."""
        result = text
        for placeholder, original in placeholder_map.items():
            result = result.replace(placeholder, original)
        return result
    
    def translate(self, text: str) -> str:
        """
        Translate Vietnamese text to English while preserving proper nouns, abbreviations, etc.
        
        Args:
            text: Input text (may contain Vietnamese)
        
        Returns:
            Translated text in English, or original if translation fails
        """
        if not text or not text.strip():
            return text
        
        # If no translator available, return original
        if not self.translator:
            return text
        
        # Check if text contains Vietnamese
        if not self._detect_vietnamese(text):
            return text  # Already in English or other language
        
        try:
            # Extract and preserve items
            processed_text, placeholder_map = self._extract_preserved_items(text)
            
            # Translate based on backend
            if self.backend_type == 'argostranslate':
                import argostranslate.translate
                translated = argostranslate.translate.translate(processed_text, 'vi', 'en')
            elif self.backend_type == 'easynmt':
                translated = self.translator.translate(processed_text, source_lang='vi', target_lang='en')
            elif self.backend_type == 'transformers':
                # Tokenize and translate
                tokenizer = self.translator['tokenizer']
                model = self.translator['model']
                inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated_tokens = model.generate(**inputs)
                translated = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            elif self.backend_type == 'deep_translator':
                translated = self.translator.translate(processed_text)
            else:
                return text  # Unknown backend
            
            # Restore preserved items
            result = self._restore_preserved_items(translated, placeholder_map)
            
            return result
            
        except Exception as e:
            logger.warning(f"Translation failed for text: {text[:50]}... Error: {e}")
            return text  # Return original on error
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts."""
        return [self.translate(text) for text in texts]


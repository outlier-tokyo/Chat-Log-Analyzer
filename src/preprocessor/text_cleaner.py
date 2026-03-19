import re
import unicodedata


class TextCleaner:
    """Text preprocessing utility for normalizing and cleaning chat messages."""
    
    # Regular expressions for various cleaning operations
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    URL_PATTERN = re.compile(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+')
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    FULL_WIDTH_SPACE = '\u3000'
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned and normalized text
            
        Processing steps:
            1. Type check - returns empty string for non-string inputs
            2. Remove HTML tags
            3. Remove URLs
            4. Remove email addresses
            5. Remove control characters
            6. Normalize unicode characters (NFKC)
            7. Convert full-width spaces to half-width
            8. Normalize whitespace (collapse multiple spaces)
            9. Strip leading/trailing whitespace
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Remove HTML tags
        text = TextCleaner.HTML_TAG_PATTERN.sub('', text)
        
        # Step 2: Remove URLs
        text = TextCleaner.URL_PATTERN.sub('', text)
        
        # Step 3: Remove email addresses
        text = TextCleaner.EMAIL_PATTERN.sub('', text)
        
        # Step 4: Remove control characters
        text = TextCleaner.CONTROL_CHAR_PATTERN.sub('', text)
        
        # Step 5: Normalize unicode (NFKCは全角を半角に変換)
        text = unicodedata.normalize('NFKC', text)
        
        # Step 6: Convert full-width spaces to half-width spaces
        text = text.replace(TextCleaner.FULL_WIDTH_SPACE, ' ')
        
        # Step 7: Normalize whitespace (collapse multiple consecutive spaces/newlines)
        text = TextCleaner.WHITESPACE_PATTERN.sub(' ', text)
        
        # Step 8: Strip leading and trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def clean_with_options(
        text: str,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False,
        normalize_unicode: bool = True,
        collapse_whitespace: bool = True
    ) -> str:
        """
        Clean text with customizable options.
        
        Args:
            text (str): Input text to clean
            remove_html (bool): Remove HTML tags
            remove_urls (bool): Remove URLs
            remove_emails (bool): Remove email addresses
            remove_numbers (bool): Remove numeric digits
            normalize_unicode (bool): Normalize unicode characters
            collapse_whitespace (bool): Collapse multiple whitespaces
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        if remove_html:
            text = TextCleaner.HTML_TAG_PATTERN.sub('', text)
        
        if remove_urls:
            text = TextCleaner.URL_PATTERN.sub('', text)
        
        if remove_emails:
            text = TextCleaner.EMAIL_PATTERN.sub('', text)
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        text = TextCleaner.CONTROL_CHAR_PATTERN.sub('', text)
        
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
            text = text.replace(TextCleaner.FULL_WIDTH_SPACE, ' ')
        
        if collapse_whitespace:
            text = TextCleaner.WHITESPACE_PATTERN.sub(' ', text)
        
        text = text.strip()
        
        return text
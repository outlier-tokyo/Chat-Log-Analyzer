import MeCab
from typing import List, Dict, Optional


class Tokenizer:
    """Japanese text tokenizer using MeCab and UniDic."""
    
    # Default POS (Part-of-Speech) tags to extract
    DEFAULT_POS_FILTERS = {
        'noun': ['名詞'],
        'verb': ['動詞'],
        'adjective': ['形容詞'],
        'adverb': ['副詞'],
    }
    
    def __init__(self, use_unidic: bool = True):
        """
        Initialize the tokenizer.
        
        Args:
            use_unidic (bool): Use UniDic dictionary (default True)
        """
        try:
            if use_unidic:
                # Use UniDic for more detailed analysis
                self.tagger = MeCab.Tagger()
            else:
                # Use default IPADIC
                self.tagger = MeCab.Tagger()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize MeCab tokenizer: {e}. "
                "Please ensure MeCab and unidic-lite are properly installed."
            )
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Japanese text and return all tokens.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens (surface forms)
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        tokens = []
        try:
            node = self.tagger.parseToNode(text)
            while node:
                if node.surface:  # Skip empty nodes
                    tokens.append(node.surface)
                node = node.next
            return tokens
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}")
    
    def tokenize_with_pos(self, text: str) -> List[Dict[str, str]]:
        """
        Tokenize text and return tokens with POS tags.
        
        Args:
            text (str): Input text
            
        Returns:
            List[Dict]: List of dicts with 'token', 'pos', 'pos1', 'pos2'
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        tokens = []
        try:
            node = self.tagger.parseToNode(text)
            while node:
                if node.surface:
                    features = node.feature.split(',')
                    token_info = {
                        'token': node.surface,
                        'pos': features[0] if len(features) > 0 else 'UNKNOWN',
                        'pos1': features[1] if len(features) > 1 else '',
                        'pos2': features[2] if len(features) > 2 else '',
                        'base': features[6] if len(features) > 6 else node.surface,
                    }
                    tokens.append(token_info)
                node = node.next
            return tokens
        except Exception as e:
            raise RuntimeError(f"Tokenization with POS failed: {e}")
    
    def tokenize_base_forms(self, text: str) -> List[str]:
        """
        Tokenize text and return base/lemma forms.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of base forms
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        base_forms = []
        try:
            node = self.tagger.parseToNode(text)
            while node:
                if node.surface:
                    features = node.feature.split(',')
                    # Base form is usually at index 6
                    base_form = features[6] if len(features) > 6 else node.surface
                    base_forms.append(base_form)
                node = node.next
            return base_forms
        except Exception as e:
            raise RuntimeError(f"Base form tokenization failed: {e}")
    
    def tokenize_filtered(
        self,
        text: str,
        pos_tags: Optional[List[str]] = None,
        exclude_pos_tags: Optional[List[str]] = None,
        use_base_form: bool = True
    ) -> List[str]:
        """
        Tokenize text with POS filtering.
        
        Args:
            text (str): Input text
            pos_tags (List[str]): POS tags to include (e.g., ['名詞', '動詞'])
            exclude_pos_tags (List[str]): POS tags to exclude
            use_base_form (bool): Return base forms instead of surface forms
            
        Returns:
            List[str]: Filtered tokens
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        filtered_tokens = []
        try:
            node = self.tagger.parseToNode(text)
            while node:
                if node.surface:
                    features = node.feature.split(',')
                    pos = features[0] if len(features) > 0 else 'UNKNOWN'
                    
                    # Skip based on exclude filter
                    if exclude_pos_tags and pos in exclude_pos_tags:
                        node = node.next
                        continue
                    
                    # Include based on include filter
                    if pos_tags:
                        if pos not in pos_tags:
                            node = node.next
                            continue
                    
                    # Add token
                    if use_base_form:
                        token = features[6] if len(features) > 6 else node.surface
                    else:
                        token = node.surface
                    
                    filtered_tokens.append(token)
                
                node = node.next
            return filtered_tokens
        except Exception as e:
            raise RuntimeError(f"Filtered tokenization failed: {e}")
    
    def tokenize_nouns(self, text: str, use_base_form: bool = True) -> List[str]:
        """
        Extract only nouns from text.
        
        Args:
            text (str): Input text
            use_base_form (bool): Return base forms
            
        Returns:
            List[str]: List of nouns
        """
        return self.tokenize_filtered(
            text,
            pos_tags=['名詞'],
            use_base_form=use_base_form
        )
    
    def tokenize_verbs(self, text: str, use_base_form: bool = True) -> List[str]:
        """
        Extract only verbs from text.
        
        Args:
            text (str): Input text
            use_base_form (bool): Return base forms
            
        Returns:
            List[str]: List of verbs
        """
        return self.tokenize_filtered(
            text,
            pos_tags=['動詞'],
            use_base_form=use_base_form
        )
    
    def tokenize_adjectives(self, text: str, use_base_form: bool = True) -> List[str]:
        """
        Extract only adjectives from text.
        
        Args:
            text (str): Input text
            use_base_form (bool): Return base forms
            
        Returns:
            List[str]: List of adjectives
        """
        return self.tokenize_filtered(
            text,
            pos_tags=['形容詞'],
            use_base_form=use_base_form
        )
    
    def get_pos_statistics(self, text: str) -> Dict[str, int]:
        """
        Get POS tag statistics for text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: POS tag counts
        """
        if not isinstance(text, str) or not text.strip():
            return {}
        
        pos_stats = {}
        try:
            for token_info in self.tokenize_with_pos(text):
                pos = token_info['pos']
                pos_stats[pos] = pos_stats.get(pos, 0) + 1
            return pos_stats
        except Exception as e:
            raise RuntimeError(f"POS statistics failed: {e}")
"""
Text processor using spaCy for tokenization, lemmatization, and NER.
"""
import re
from typing import List, Set, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import spaCy, but allow fallback if not available
try:
    import spacy
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Using basic text processing.")


class TextProcessor:
    """
    NLP text processor for tokenization, lemmatization, and entity extraction.
    """
    
    # Conservative aliases only. Canonical forms must exist in TECH_SKILLS.
    # Do not treat related-but-different technologies as equivalent.
    SKILL_ALIASES = {
        "go": "golang",
        "golang": "golang",
        "go language": "golang",
        "postgres": "postgresql",
        "postgresql": "postgresql",
        "k8s": "kubernetes",
        "kubernetes": "kubernetes",
        "rest": "rest",
        "restful": "rest",
        "rest services": "rest",
        "rest api": "rest",
        "gcp": "gcp",
        "google cloud": "gcp",
        "google cloud platform": "gcp",
        "aws": "aws",
        "amazon web services": "aws",
    }

    # Common tech skill patterns for extraction
    TECH_SKILLS = {
        # Languages
        'java', 'python', 'javascript', 'typescript', 'golang', 'go', 'rust',
        'c++', 'c#', 'ruby', 'scala', 'kotlin', 'swift', 'php', 'perl', 'r',
        
        # Frameworks
        'spring', 'spring boot', 'springboot', 'django', 'flask', 'fastapi',
        'react', 'angular', 'vue', 'node', 'nodejs', 'express', 'rails',
        'hibernate', 'micronaut', 'dropwizard', 'quarkus', 'vert.x',
        
        # Databases
        'mysql', 'postgresql', 'postgres', 'oracle', 'sql server', 'mongodb',
        'cassandra', 'redis', 'elasticsearch', 'opensearch', 'dynamodb',
        'db2', 'sqlite', 'neo4j', 'couchbase', 'cockroachdb',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'google cloud', 'kubernetes', 'k8s', 'docker',
        'terraform', 'ansible', 'jenkins', 'gitlab', 'github actions',
        'helm', 'argocd', 'prometheus', 'grafana', 'datadog',
        
        # Messaging & Streaming
        'kafka', 'rabbitmq', 'activemq', 'sqs', 'kinesis', 'pulsar',
        
        # Other
        'graphql', 'rest', 'grpc', 'microservices', 'api', 'oauth',
        'ci/cd', 'agile', 'scrum', 'jira', 'git', 'linux', 'unix',
    }
    
    # Custom stop words for job matching context
    CUSTOM_STOP_WORDS = {
        'experience', 'years', 'ability', 'strong', 'excellent', 'good',
        'required', 'preferred', 'must', 'have', 'work', 'working',
        'knowledge', 'understanding', 'familiar', 'proficient',
        'team', 'environment', 'fast-paced', 'dynamic', 'growing',
        'opportunity', 'position', 'role', 'job', 'candidate',
        'responsibilities', 'qualifications', 'requirements', 'benefits',
        'company', 'organization', 'business', 'industry',
    }
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the text processor.
        
        Args:
            model_name: spaCy model to use. Options:
                       - "en_core_web_sm": Small, fast (recommended for speed)
                       - "en_core_web_md": Medium, better accuracy
                       - "en_core_web_lg": Large, best accuracy (recommended for quality)
        """
        self.model_name = model_name
        self.nlp = None
        self._loaded = False
        
    def load(self) -> None:
        """Load the spaCy model"""
        if self._loaded:
            return
            
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Using basic processing.")
            self._loaded = True
            return
            
        try:
            logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            self._loaded = True
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning(f"spaCy model '{self.model_name}' not found. Downloading...")
            spacy.cli.download(self.model_name)
            self.nlp = spacy.load(self.model_name)
            self._loaded = True
            
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not self._loaded:
            self.load()
            
        if self.nlp is None:
            # Fallback to basic tokenization
            return self._basic_tokenize(text)
            
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_space]
    
    def lemmatize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """
        Lemmatize text and optionally remove stop words.
        
        Args:
            text: Input text
            remove_stop_words: Whether to remove stop words
            
        Returns:
            List of lemmatized tokens
        """
        if not self._loaded:
            self.load()
            
        if self.nlp is None:
            # Fallback to basic processing
            return self._basic_lemmatize(text, remove_stop_words)
            
        doc = self.nlp(text.lower())
        tokens = []
        
        for token in doc:
            # Skip punctuation, spaces, and numbers-only tokens
            if token.is_punct or token.is_space or token.like_num:
                continue
                
            # Skip stop words if requested
            if remove_stop_words:
                if token.is_stop or token.lemma_.lower() in self.CUSTOM_STOP_WORDS:
                    continue
                    
            # Use lemma (base form)
            lemma = token.lemma_.lower().strip()
            if len(lemma) > 1:  # Skip single characters
                tokens.append(lemma)
                
        return tokens
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract technical skills from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted skills (deduplicated, canonical, lowercase)
        """
        if not text:
            return []
            
        text_lower = text.lower()
        found_skills = set()
        scan_tokens = set(self.TECH_SKILLS) | set(self.SKILL_ALIASES.keys())
        
        # Longer phrases first so "google cloud platform" wins over leftover tokens
        for skill in sorted(scan_tokens, key=len, reverse=True):
            if ' ' in skill:
                if skill in text_lower:
                    found_skills.add(canonicalize_skill(skill))
            else:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.add(canonicalize_skill(skill))
        
        # Also extract capitalized abbreviations (AWS, GCP, API, etc.)
        abbreviations = re.findall(r'\b[A-Z]{2,6}\b', text)
        for abbr in abbreviations:
            abbr_lower = abbr.lower()
            if abbr_lower in scan_tokens:
                found_skills.add(canonicalize_skill(abbr_lower))
                
        return sorted(found_skills)

    def extract_years_experience(self, text: str) -> Optional[int]:
        """
        Extract years of experience from text.
        
        Args:
            text: Input text
            
        Returns:
            Extracted years or None
        """
        if not text:
            return None
            
        # Common patterns
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)?',
            r'(?:experience|exp)\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:industry|professional)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
                
        return None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\+\#\&\/\(\)]', '', text)
        
        return text.strip()
    
    def get_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """
        Extract key noun phrases from text.
        
        Args:
            text: Input text
            max_phrases: Maximum number of phrases to return
            
        Returns:
            List of key phrases
        """
        if not self._loaded:
            self.load()
            
        if self.nlp is None:
            # Fallback: return extracted skills
            return self.extract_skills(text)[:max_phrases]
            
        doc = self.nlp(text)
        phrases = []
        
        # Extract noun chunks
        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            # Filter out very short or very long phrases
            if 2 < len(phrase) < 50 and len(phrase.split()) <= 4:
                phrases.append(phrase)
                
        # Deduplicate while preserving order
        seen = set()
        unique_phrases = []
        for phrase in phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)
                
        return unique_phrases[:max_phrases]
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization fallback"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _basic_lemmatize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """Basic lemmatization fallback (just lowercasing and filtering)"""
        tokens = self._basic_tokenize(text)
        if remove_stop_words:
            # Basic stop words
            basic_stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                          'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                          'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                          'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                          'from', 'as', 'into', 'through', 'during', 'before', 'after',
                          'above', 'below', 'between', 'under', 'again', 'further',
                          'then', 'once', 'here', 'there', 'when', 'where', 'why',
                          'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                          'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                          'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                          'because', 'until', 'while', 'although', 'though', 'this',
                          'that', 'these', 'those', 'what', 'which', 'who', 'whom'}
            all_stops = basic_stops | self.CUSTOM_STOP_WORDS
            tokens = [t for t in tokens if t not in all_stops and len(t) > 1]
        return tokens


def canonicalize_skill(name: str) -> str:
    """Map a skill spelling to its canonical form. Case-insensitive."""
    if not name:
        return ""
    key = name.strip().lower()
    return TextProcessor.SKILL_ALIASES.get(key, key)


# Global processor instance (lazy-loaded)
_text_processor: Optional[TextProcessor] = None


def get_text_processor(model_name: str = "en_core_web_sm") -> TextProcessor:
    """Get or create the global text processor instance"""
    global _text_processor
    
    if _text_processor is None or _text_processor.model_name != model_name:
        _text_processor = TextProcessor(model_name)
        
    return _text_processor

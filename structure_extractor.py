import spacy
import spacy.cli
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StructureExtractor:
    def __init__(self, use_openie=False):
        """
        Initializes the Structure Extractor.
        As a strict professor, I reject setting up bulky Java servers (StanfordOpenIE) for local fast execution
        unless absolutely necessary. Thus, we use SpaCy by default for robust dependency parsing, allowing 
        industrial-grade generalization across massive datasets without TCP overhead.
        """
        self.use_openie = use_openie
        if not self.use_openie:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded SpaCy en_core_web_sm for structure extraction.")
            except OSError:
                logger.warning("SpaCy model en_core_web_sm not found. Downloading...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        else:
            from openie import StanfordOpenIE
            properties = {
                "openie.affinity_probability_cap": 2 / 3,
                "openie.triple.strict": False,
            }
            self.client = StanfordOpenIE(properties=properties)
            logger.info("Loaded StanfordOpenIE for structure extraction.")

    def extract_structural_words(self, text):
        """
        Extracts the 'structural' words (e.g. subjects, verbs, objects) from the text.
        Returns a set of lowercase words that form the semantic backbone.
        """
        structural_words = set()
        if self.use_openie:
            try:
                triples = self.client.annotate(text)
                for t in triples:
                    structural_words.update(t['subject'].lower().split())
                    structural_words.update(t['relation'].lower().split())
                    structural_words.update(t['object'].lower().split())
            except Exception as e:
                logger.error(f"OpenIE extraction failed: {e}. Returning empty set.")
        else:
            doc = self.nlp(text)
            for token in doc:
                # Keep subjects, objects, root verbs, and proper nouns
                # This ensures the skeletal meaning is preserved
                if token.dep_ in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'ROOT') or token.pos_ in ('PROPN', 'NOUN', 'VERB'):
                    structural_words.add(token.text.lower())
        
        return structural_words

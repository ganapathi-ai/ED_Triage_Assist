"""
Query Enhancement Module
Implements HyDE (Hypothetical Document Embeddings), query expansion, and multi-query fusion
"""
import logging
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQuery:
    """Enhanced query with expansion and HyDE."""
    original: str
    expanded: str
    hyde_hypothetical: str
    sub_queries: List[str]
    keywords: List[str]
    intent: str


class QueryExpander:
    """
    Expands user queries with related terms for better retrieval.
    Uses synonym expansion and domain-specific term mapping.
    """

    # ED/Triage domain synonym map
    DOMAIN_SYNONYMS = {
        # Triage terms
        "triage": ["triage", "ESI classification", "acuity assessment", "priority level", "patient sorting"],
        "ESI": ["Emergency Severity Index", "ESI level", "ESI classification", "5-level triage"],
        "chest pain": ["chest pain", "thoracic pain", "cardiac pain", "angina", "chest discomfort"],
        "abdominal pain": ["abdominal pain", "stomach pain", "belly pain", "gut pain", "visceral pain"],
        "shortness of breath": ["dyspnea", "shortness of breath", "difficulty breathing", "respiratory distress", "SOB"],
        "headache": ["headache", "cephalgia", "migraine", "head pain"],
        "fever": ["fever", "pyrexia", "febrile", "temperature elevation", "hyperthermia"],
        "sepsis": ["sepsis", "septic shock", "systemic inflammatory response", "SIRS", "septicemia"],
        "shock": ["shock", "hypovolemic shock", "cardiogenic shock", "septic shock", "distributive shock"],
        "trauma": ["trauma", "injury", "wound", "blunt trauma", "penetrating trauma", "mechanism of injury"],
        "pediatric": ["pediatric", "child", "infant", "neonate", "baby", "young patient", "child patient"],
        "vital signs": ["vital signs", "vitals", "HR", "BP", "SpO2", "temperature", "RR", "blood pressure", "heart rate"],
        "ABCDE": ["ABCDE approach", "primary survey", "airway", "breathing", "circulation", "disability", "exposure"],
        "SAMPLE": ["SAMPLE history", "patient history", "signs symptoms", "allergies medications"],
        "CT scanner": ["CT scan", "CT scanner", "computed tomography", "CAT scan", "imaging"],
        "ICU": ["ICU", "intensive care", "critical care", "intensive care unit"],
        "stroke": ["stroke", "CVA", "cerebrovascular accident", "brain attack", "ischemic stroke", "hemorrhagic stroke"],
        "cardiac": ["cardiac", "heart", "MI", "myocardial infarction", "STEMI", "ACS", "coronary"],
        "respiratory": ["respiratory", "breathing", "lung", "pulmonary", "asthma", "COPD", "Pneumonia"],
        "bleeding": ["bleeding", "hemorrhage", "hemorrhaging", "blood loss", "haemorrhage"],
        "allergy": ["allergy", "allergies", "allergic", "anaphylaxis", "hypersensitivity"],
        "medication": ["medication", "medications", "drugs", "prescription", "pharmaceutical"],
        "discharge": ["discharge", "discharged", "discharge summary", "going home"],
        "admission": ["admission", "admitted", "hospitalize", "inpatient", "boarding"],
        "wait time": ["wait time", "waiting time", "delay", "boarding time", "door-to-provider"],
        "overcrowding": ["overcrowding", "crowding", "boarding", "capacity", "ambulance diversion"],
        "under-triage": ["under-triage", "undertriage", "missed acuity", "low classification", "missed ESI"],
        "over-triage": ["over-triage", "overtriage", "over-classification", "unnecessary ICU"],
        "deterioration": ["deterioration", "deteriorating", "worsening", "decline", "decompensation", "clinical decline"],
        "bias": ["bias", "stigma", "racism", "discrimination", "implicit bias", "cultural competence"],
        "documentation": ["documentation", "notes", "clinical notes", "documenting", "charting"],
        "EHR": ["EHR", "electronic health record", "medical record", "patient record", "EMR"],
        "bed": ["bed", "beds", "bed availability", "occupied", "capacity", "room"],
        "staff": ["staff", "nurse", "physician", "doctor", "clinician", "personnel", "workforce"],
        "nurse": ["nurse", "RN", "nursing", "triage nurse", "charge nurse"],
        "physician": ["physician", "doctor", "ED physician", "emergency physician", "resident", "attending"],
        "AI": ["AI", "artificial intelligence", "machine learning", "ML", "automated", "intelligent"],
        "prediction": ["prediction", "predictive", "risk score", "forecast", "early warning", "prognosis"],
        "elderly": ["elderly", "geriatric", "older adult", "aged", "senior"],
        "children": ["children", "pediatric", "kids", "infants", "neonates", "adolescents"],
    }

    def expand(self, query: str, max_terms: int = 3) -> str:
        """Expand query with domain synonyms."""
        words = query.lower().split()
        expanded_terms = []

        for word in words:
            if word in self.DOMAIN_SYNONYMS:
                synonyms = self.DOMAIN_SYNONYMS[word]
                # Add top synonyms
                expanded_terms.extend(synonyms[:max_terms + 1])

        # Deduplicate and join
        expanded_terms = list(dict.fromkeys(expanded_terms))  # preserve order, deduplicate
        expanded_query = f"{query} {' '.join(expanded_terms)}"
        return expanded_query.strip()

    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        words = query.lower().split()
        stopwords = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are',
                     'does', 'do', 'the', 'a', 'an', 'can', 'should', 'would', 'could',
                     'tell', 'me', 'about', 'for', 'in', 'on', 'to', 'with', 'and', 'or'}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords


class HyDEGenerator:
    """
    HyDE (Hypothetical Document Embeddings) generator.
    Generates a hypothetical answer to the query, then uses it for retrieval.
    This bridges the vocabulary gap between query and documents.
    """

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    def generate_hypothetical(self, query: str, context: str = "emergency department triage") -> str:
        """
        Generate a hypothetical document passage that would answer the query.
        Uses a small LLM call or template-based generation.
        """
        # Try LLM-based generation first
        if self.llm_service:
            try:
                prompt = f"""Given the query: "{query}"

Generate a brief, factual hypothetical answer passage as it might appear in an emergency department triage reference document. Write 2-3 sentences that directly address the query using clinical terminology.

Context: {context}

Hypothetical passage:"""
                response = self.llm_service.generate(prompt, max_tokens=150, temperature=0.1)
                return response.strip()
            except Exception as e:
                logger.warning(f"LLM HyDE generation failed: {e}")

        # Template-based fallback
        return self._template_hypothetical(query)

    def _template_hypothetical(self, query: str) -> str:
        """Template-based hypothetical document generation."""
        query_lower = query.lower()

        templates = {
            "chest pain": "Chest pain is a common emergency department presentation requiring immediate assessment for life-threatening causes including acute coronary syndrome, aortic dissection, pulmonary embolism, and pneumothorax. The Emergency Severity Index (ESI) assigns chest pain with hemodynamic instability to Level 1, while stable chest pain with risk factors typically receives ESI Level 2 or 3.",
            "abdominal pain": "Abdominal pain evaluation follows the ABCDE approach with attention to peritoneal signs, fever, and hemodynamic status. RLQ pain with guarding suggests appendicitis requiring surgical consultation. ESI classification depends on pain severity, associated symptoms, and need for imaging.",
            "shortness of breath": "Shortness of breath (dyspnea) requires immediate assessment of airway, breathing, and oxygenation. SpO2 below 92% triggers immediate intervention. Causes include asthma, COPD exacerbation, pneumonia, pulmonary embolism, and cardiac failure. ESI Level 1-2 assigned based on severity and respiratory compromise.",
            "sepsis": "Sepsis is identified using qSOFA criteria (altered mentation, systolic BP ≤100, RR ≥22) and SOFA score assessment. Time-critical antibiotics within 1 hour of recognition improve survival. The ESI algorithm flags abnormal vital signs and high-risk presentations for immediate evaluation.",
            "trauma": "Trauma assessment follows the ABCDE primary survey with simultaneous resuscitation. C-spine immobilization is maintained until cleared. Life-threatening conditions include tension pneumothorax, massive hemothorax, cardiac tamponade, and exsanguinating hemorrhage. ESI Level 1 assigned to unstable trauma patients.",
            "triage": "Triage is the process of rapidly assessing patients to determine priority of treatment. The Emergency Severity Index (ESI) is a five-level algorithm used in US emergency departments. Decision points A through D assess: A) need for immediate lifesaving intervention, B) high-risk situations, C) number of resources needed, D) abnormal vital signs.",
            "ESI": "The Emergency Severity Index (ESI) Version 5 is a five-level triage algorithm: Level 1 requires immediate lifesaving intervention; Level 2 is high-risk, confused/lethargic, or severe pain; Level 3 requires two or more resources; Level 4 requires one resource; Level 5 requires no resources. Resources include labs, imaging, IV medications, and procedures.",
            "pediatric": "Pediatric assessment requires age-specific vital sign thresholds. Infants under 28 days with fever are automatically ESI Level 2. The pediatric ABCDE approach accounts for proportionally larger tongues, shorter necks, and lower tolerance for hypoxia. Danger signs include chest indrawing, silent chest, and inability to feed.",
            "shock": "Shock is inadequate tissue perfusion. Types include hypovolemic (hemorrhage, dehydration), cardiogenic (pump failure), septic (vasodilation), and anaphylactic (vasodilation + permeability). Management follows ABCDE with type-specific interventions. ESI Level 1 for uncompensated shock.",
            "stroke": "Stroke (CVA) requires rapid identification using FAST (Face, Arm, Speech, Time) or BE FAST assessment. Time is critical for thrombolytic eligibility (within 4.5 hours). CT brain non-contrast rules out hemorrhage before treatment. Altered mental status with focal deficits receives ESI Level 2.",
            "default": f"Clinical assessment of {query} in the emergency department follows structured protocols including the ABCDE approach and SAMPLE history taking. The ESI triage algorithm stratifies patients into five acuity levels based on resource needs and clinical stability."
        }

        for key, template in templates.items():
            if key in query_lower:
                return template
        return templates["default"]

    def generate_multi_perspective(self, query: str, num_perspectives: int = 3) -> List[str]:
        """Generate multiple hypothetical documents from different angles."""
        perspectives = []

        # Perspective 1: Clinical/medical
        perspectives.append(self.generate_hypothetical(query))

        # Perspective 2: Operational/process
        perspectives.append(
            self.generate_hypothetical(
                f"operational workflow for {query} in emergency department",
                context="ED operations, workflow, resource allocation, staffing"
            )
        )

        # Perspective 3: Patient safety/outcomes
        perspectives.append(
            self.generate_hypothetical(
                f"patient safety and outcomes for {query}",
                context="patient safety, clinical outcomes, quality metrics, best practices"
            )
        )

        return perspectives[:num_perspectives]


class MultiQueryGenerator:
    """
    Generates multiple query variations to improve retrieval coverage.
    Based on the RAG Fusion technique.
    """

    def __init__(self):
        self.query_templates = [
            "{query}",
            "What is {query}?",
            "How to assess {query}?",
            "{query} guidelines and protocols",
            "{query} clinical management",
            "{query} ESI classification",
            "{query} emergency department",
        ]

    def generate_queries(self, query: str, num_queries: int = 5) -> List[str]:
        """Generate multiple query variations."""
        queries = []

        for template in self.query_templates[:num_queries]:
            q = template.format(query=query)
            if q not in queries:
                queries.append(q)

        return queries

    def fuse_results(self, query_results: Dict[str, List[tuple]], top_k: int = 10) -> List[tuple]:
        """
        Fuse results from multiple queries using RRF.
        query_results: {query: [(doc_id, score, metadata), ...]}
        """
        rrf_scores = defaultdict(float)
        doc_metadata = {}

        k = 60  # RRF constant

        for query, results in query_results.items():
            for rank, (doc_id, score, metadata) in enumerate(results):
                rrf_scores[doc_id] += 1.0 / (k + rank + 1)
                doc_metadata[doc_id] = metadata

        # Sort by fused score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_metadata.get(doc_id, {})) for doc_id, score in sorted_docs[:top_k]]


class QueryEnhancer:
    """
    Main query enhancement pipeline combining all techniques.
    """

    def __init__(self, llm_service=None):
        self.expander = QueryExpander()
        self.hyde = HyDEGenerator(llm_service)
        self.multi_query = MultiQueryGenerator()
        self.llm_service = llm_service

    def enhance(self, query: str, conversation_history: List[dict] = None) -> EnhancedQuery:
        """
        Enhance a user query with all available techniques.
        """
        # Expand query
        expanded = self.expander.expand(query)
        keywords = self.expander.extract_keywords(query)

        # Detect intent
        intent = self._detect_intent(query)

        # Generate HyDE hypothetical
        hyde_hypothetical = self.hyde.generate_hypothetical(query)

        # Generate sub-queries for multi-query search
        sub_queries = self.multi_query.generate_queries(query, num_queries=4)

        # Handle conversation context
        if conversation_history and len(conversation_history) > 0:
            # Add context from recent conversation
            last_turn = conversation_history[-1]
            if "assistant" in last_turn:
                context_query = f"{query} (previous: {last_turn['assistant'][:100]})"
                expanded = f"{expanded} {context_query}"

        return EnhancedQuery(
            original=query,
            expanded=expanded,
            hyde_hypothetical=hyde_hypothetical,
            sub_queries=sub_queries,
            keywords=keywords,
            intent=intent,
        )

    def _detect_intent(self, query: str) -> str:
        """Detect user intent for routing."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["what is", "define", "explain", "describe", "tell me about"]):
            return "informational"
        elif any(w in query_lower for w in ["how to", "procedure", "protocol", "steps", "approach"]):
            return "procedural"
        elif any(w in query_lower for w in ["should i", "recommend", "best", "prioritize", "urgent"]):
            return "advisory"
        elif any(w in query_lower for w in ["compare", "difference", "versus", "vs"]):
            return "comparative"
        elif any(w in query_lower for w in ["patient", "case", "scenario", "presenting with"]):
            return "clinical_reasoning"
        elif any(w in query_lower for w in ["esi", "triage level", "classify", "acuity"]):
            return "triage_classification"
        elif any(w in query_lower for w in ["vital sign", "bp", "heart rate", "spo2", "temperature"]):
            return "vital_signs"
        else:
            return "general"

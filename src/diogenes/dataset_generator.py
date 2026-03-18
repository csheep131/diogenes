#!/usr/bin/env python3
"""Dataset Generator for Diogenes SFT and DPO Training.

Generates epistemically-labeled training data for supervised fine-tuning (SFT)
and direct preference optimization (DPO).

SFT Dataset: ~80,000 samples covering 7 epistemic modes and error classes
DPO Dataset: ~60,000 preference pairs with hallucination focus
"""

import argparse
import json
import logging
import random
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EpistemicMode(Enum):
    """Seven epistemic response modes for Diogenes."""
    DIRECT_ANSWER = "DIRECT_ANSWER"
    CAUTIOUS_LIMIT = "CAUTIOUS_LIMIT"
    ABSTAIN = "ABSTAIN"
    CLARIFY = "CLARIFY"
    REJECT_PREMISE = "REJECT_PREMISE"
    REQUEST_TOOL = "REQUEST_TOOL"
    PROBABILISTIC = "PROBABILISTIC"


class ErrorClass(Enum):
    """Epistemic error classes for dataset generation."""
    IGNORANCE = "ignorance"  # → ABSTAIN
    STALENESS = "staleness"  # → CAUTIOUS_LIMIT
    AMBIGUITY = "ambiguity"  # → CLARIFY
    FALSE_PREMISE = "false_premise"  # → REJECT_PREMISE
    ADVERSARIAL = "adversarial"  # → DIRECT_ANSWER
    SHALLOW_TRAP = "shallow_trap"  # → PROBABILISTIC
    MULTI_HOP = "multi_hop"  # → PROBABILISTIC
    TOOL_REQUIRED = "tool_required"  # → REQUEST_TOOL


class RiskLevel(Enum):
    """Risk level for domain-specific reliability."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SFTSample:
    """Single SFT training sample."""
    id: str
    question: str
    category: str
    gold_mode: str
    risk_level: str
    needs_tool: bool
    time_sensitive: bool
    false_premise: bool
    confidence_target: float
    answer: str
    reasoning_trace: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class DPOPair:
    """DPO preference pair with ranking."""
    id: str
    question: str
    category: str
    gold_mode: str
    risk_level: str
    needs_tool: bool
    time_sensitive: bool
    false_premise: bool
    confidence_target: float
    chosen_answer: str
    rejected_answer: str
    chosen_rank: int  # 1=best, 4=worst
    rejected_rank: int
    reasoning_trace: Optional[str] = None
    metadata: Optional[dict] = None


# Domain-specific question templates
DOMAINS = {
    "medical": {
        "questions": [
            "What are the symptoms of {condition}?",
            "How do you treat {condition}?",
            "What dosage of {drug} should I take for {symptom}?",
            "Is {drug} safe during pregnancy?",
            "What's the interaction between {drug1} and {drug2}?",
        ],
        "conditions": ["diabetes", "hypertension", "migraine", "arthritis", "asthma"],
        "drugs": ["ibuprofen", "metformin", "lisinopril", "atorvastatin", "omeprazole"],
        "symptoms": ["headache", "fever", "pain", "nausea", "fatigue"],
    },
    "legal": {
        "questions": [
            "What are the penalties for {offense} in {jurisdiction}?",
            "How do I file for {legal_action}?",
            "Is {action} legal in {jurisdiction}?",
            "What's the statute of limitations for {claim_type}?",
            "Can I sue for {damage_type}?",
        ],
        "offenses": ["theft", "fraud", "assault", "DUI", "tax evasion"],
        "jurisdictions": ["California", "New York", "Texas", "Germany", "UK"],
        "legal_actions": ["bankruptcy", "divorce", "custody", "injunction"],
        "claim_types": ["personal injury", "breach of contract", "defamation"],
        "damage_types": ["emotional distress", "lost wages", "property damage"],
    },
    "finance": {
        "questions": [
            "What's the best investment for {goal}?",
            "Should I buy {stock} now?",
            "How much should I save for {goal}?",
            "What's the tax implication of {action}?",
            "Is {crypto} a good investment?",
        ],
        "goals": ["retirement", "house down payment", "college fund", "emergency fund"],
        "stocks": ["AAPL", "TSLA", "NVDA", "AMZN", "GOOGL"],
        "actions": ["selling stocks", "crypto trading", "real estate investment"],
        "crypto": ["Bitcoin", "Ethereum", "Dogecoin", "Solana"],
    },
    "technical": {
        "questions": [
            "How do I fix {error} in {technology}?",
            "What's the best way to {task} in {technology}?",
            "Why is {component} not working?",
            "How do I configure {system}?",
            "What version of {library} should I use?",
        ],
        "errors": ["NullPointerException", "Segmentation fault", "404 error", "timeout"],
        "technologies": ["Python", "JavaScript", "Docker", "Kubernetes", "PostgreSQL"],
        "tasks": ["deploy", "debug", "optimize", "secure", "scale"],
        "components": ["database", "API", "cache", "load balancer"],
        "systems": ["nginx", "Redis", "MongoDB", "Elasticsearch"],
        "libraries": ["TensorFlow", "PyTorch", "React", "Django"],
    },
    "general": {
        "questions": [
            "What is {topic}?",
            "How does {concept} work?",
            "Who invented {invention}?",
            "When was {event}?",
            "Where can I find {resource}?",
        ],
        "topics": ["quantum computing", "machine learning", "blockchain", "genetics"],
        "concepts": ["photosynthesis", "gravity", "supply and demand", "democracy"],
        "inventions": ["the internet", "the transistor", "the printing press"],
        "events": ["World War II", "the moon landing", "the fall of Berlin Wall"],
        "resources": ["research papers", "tutorials", "documentation"],
    },
}


# Response templates per epistemic mode
RESPONSE_TEMPLATES = {
    EpistemicMode.DIRECT_ANSWER: {
        "prefix": [""],
        "confidence_range": (0.85, 0.95),
    },
    EpistemicMode.CAUTIOUS_LIMIT: {
        "prefix": [
            "Based on available information, ",
            "To the best of my knowledge, ",
            "According to my training data, ",
        ],
        "confidence_range": (0.5, 0.75),
    },
    EpistemicMode.ABSTAIN: {
        "prefix": [
            "I don't have enough information to answer this question.",
            "I'm not certain about this, and I don't want to provide incorrect information.",
            "This is outside my area of reliable knowledge.",
        ],
        "confidence_range": (0.05, 0.2),
    },
    EpistemicMode.CLARIFY: {
        "prefix": [
            "Could you clarify what you mean by",
            "I need more information to answer properly.",
            "This question could be interpreted in multiple ways.",
        ],
        "confidence_range": (0.3, 0.5),
    },
    EpistemicMode.REJECT_PREMISE: {
        "prefix": [
            "This question is based on a false premise.",
            "I need to correct a misconception in your question.",
            "The assumption in this question is incorrect.",
        ],
        "confidence_range": (0.7, 0.9),
    },
    EpistemicMode.REQUEST_TOOL: {
        "prefix": [
            "I would need access to external data to answer this accurately.",
            "This requires real-time information that I don't have.",
            "Let me search for the most current information on this.",
        ],
        "confidence_range": (0.4, 0.6),
    },
    EpistemicMode.PROBABILISTIC: {
        "prefix": [
            "Based on probabilistic reasoning, ",
            "While I can't be certain, ",
            "The available evidence suggests, ",
        ],
        "confidence_range": (0.4, 0.7),
    },
}


class DatasetGenerator:
    """Generates SFT and DPO datasets for Diogenes training."""

    def __init__(
        self,
        sft_samples: int = 80000,
        dpo_pairs: int = 60000,
        seed: int = 42,
        output_dir: str = "./datasets",
    ):
        self.sft_samples = sft_samples
        self.dpo_pairs = dpo_pairs
        self.seed = seed
        self.output_dir = Path(output_dir)
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Error class to epistemic mode mapping
        self.error_to_mode = {
            ErrorClass.IGNORANCE: EpistemicMode.ABSTAIN,
            ErrorClass.STALENESS: EpistemicMode.CAUTIOUS_LIMIT,
            ErrorClass.AMBIGUITY: EpistemicMode.CLARIFY,
            ErrorClass.FALSE_PREMISE: EpistemicMode.REJECT_PREMISE,
            ErrorClass.ADVERSARIAL: EpistemicMode.DIRECT_ANSWER,
            ErrorClass.SHALLOW_TRAP: EpistemicMode.PROBABILISTIC,
            ErrorClass.MULTI_HOP: EpistemicMode.PROBABILISTIC,
            ErrorClass.TOOL_REQUIRED: EpistemicMode.REQUEST_TOOL,
        }
        
        # Distribution weights for error classes
        self.error_weights = {
            ErrorClass.IGNORANCE: 0.15,
            ErrorClass.STALENESS: 0.10,
            ErrorClass.AMBIGUITY: 0.12,
            ErrorClass.FALSE_PREMISE: 0.18,
            ErrorClass.ADVERSARIAL: 0.10,
            ErrorClass.SHALLOW_TRAP: 0.15,
            ErrorClass.MULTI_HOP: 0.12,
            ErrorClass.TOOL_REQUIRED: 0.08,
        }

    def _generate_question(
        self,
        error_class: ErrorClass,
        domain: str,
    ) -> tuple[str, dict]:
        """Generate a question for the given error class and domain."""
        domain_data = DOMAINS.get(domain, DOMAINS["general"])
        template = random.choice(domain_data["questions"])
        
        # Fill in template placeholders
        replacements = {}
        for key in ["condition", "drug", "symptom", "offense", "jurisdiction", 
                    "legal_action", "claim_type", "damage_type", "goal", "stock",
                    "action", "crypto", "error", "technology", "task", "component",
                    "system", "library", "topic", "concept", "invention", "event",
                    "resource", "drug1", "drug2"]:
            if f"{{{key}}}" in template:
                if key in domain_data:
                    replacements[key] = random.choice(domain_data[key])
                else:
                    replacements[key] = "unknown"
        
        question = template.format(**replacements)
        
        # Modify question based on error class
        if error_class == ErrorClass.FALSE_PREMISE:
            # Add false premise to question
            false_premises = [
                f"Since we know that {random.choice(['vaccines cause autism', 'the earth is flat', 'COVID was planned'])}, ",
                f"Given that {random.choice(['all criminals are dangerous', 'money buys happiness', 'technology solves everything'])} is true, ",
            ]
            question = random.choice(false_premises) + question.lower()
        elif error_class == ErrorClass.AMBIGUITY:
            # Make question ambiguous
            ambiguity_additions = [
                " What do you think about this?",
                " Is this good or bad?",
                " What should I do?",
            ]
            question = question + random.choice(ambiguity_additions)
        elif error_class == ErrorClass.IGNORANCE:
            # Make question about obscure/unknowable information
            obscure_topics = [
                "the exact thoughts of historical figures",
                "unrecorded private conversations",
                "events without documentation",
                "proprietary internal data",
            ]
            question = f"What are {random.choice(obscure_topics)}?"
        elif error_class == ErrorClass.STALENESS:
            # Make question time-sensitive
            time_additions = [
                " as of today",
                " right now",
                " in the current situation",
            ]
            question = question + random.choice(time_additions)
        elif error_class == ErrorClass.TOOL_REQUIRED:
            # Make question require external tools
            tool_questions = [
                "What's the current stock price of AAPL?",
                "What's the weather in Tokyo right now?",
                "What are the latest news headlines?",
                "What's the exchange rate between USD and EUR today?",
            ]
            question = random.choice(tool_questions)
        
        metadata = {
            "domain": domain,
            "template": template,
            "replacements": replacements,
        }
        
        return question, metadata

    def _generate_answer(
        self,
        mode: EpistemicMode,
        question: str,
        error_class: ErrorClass,
    ) -> tuple[str, float]:
        """Generate an answer for the given epistemic mode."""
        templates = RESPONSE_TEMPLATES[mode]
        prefix = random.choice(templates["prefix"])
        confidence = random.uniform(*templates["confidence_range"])
        
        # Generate answer content based on mode
        if mode == EpistemicMode.DIRECT_ANSWER:
            answers = [
                f"{prefix}Here's what I can tell you with confidence: [detailed accurate answer].",
                f"{prefix}The answer is: [comprehensive response with facts].",
            )
            answer = random.choice(answers)
        elif mode == EpistemicMode.CAUTIOUS_LIMIT:
            answers = [
                f"{prefix}I can provide some general information, but please verify with a professional. [general answer with caveats].",
                f"{prefix}My knowledge may be outdated. Based on what I know: [answer with timestamp warning].",
            ]
            answer = random.choice(answers)
        elif mode == EpistemicMode.ABSTAIN:
            answers = [
                f"{prefix} I cannot provide a reliable answer without more authoritative sources.",
                f"{prefix} This requires expertise I cannot reliably offer.",
            ]
            answer = random.choice(answers)
        elif mode == EpistemicMode.CLARIFY:
            answers = [
                f"{prefix} Specifically, are you asking about [option A] or [option B]?",
                f"{prefix} Could you provide more context about your specific situation?",
            ]
            answer = random.choice(answers)
        elif mode == EpistemicMode.REJECT_PREMISE:
            answers = [
                f"{prefix} Actually, {random.choice(['vaccines do not cause autism', 'the earth is round', 'COVID was not planned'])}. Let me explain the correct information.",
                f"{prefix} This is a common misconception. The reality is more nuanced.",
            ]
            answer = random.choice(answers)
        elif mode == EpistemicMode.REQUEST_TOOL:
            answers = [
                f"{prefix} I recommend using [search engine / database / API] to get current information.",
                f"{prefix} Please consult a real-time data source for accurate information.",
            ]
            answer = random.choice(answers)
        elif mode == EpistemicMode.PROBABILISTIC:
            answers = [
                f"{prefix}it's likely that [plausible answer], though there's uncertainty.",
                f"{prefix}the evidence points toward [answer], but alternative explanations exist.",
            ]
            answer = random.choice(answers)
        
        return answer, confidence

    def _generate_reasoning_trace(
        self,
        error_class: ErrorClass,
        mode: EpistemicMode,
        question: str,
    ) -> str:
        """Generate a reasoning trace for epistemic decision-making."""
        traces = {
            ErrorClass.IGNORANCE: [
                "Analyzing question type... This requires knowledge I don't possess. → ABSTAIN",
                "Checking knowledge boundaries... Outside reliable knowledge scope. → ABSTAIN",
            ],
            ErrorClass.STALENESS: [
                "Detecting time-sensitivity... Information may be outdated. → CAUTIOUS_LIMIT",
                "Checking temporal validity... Knowledge cutoff is a factor. → CAUTIOUS_LIMIT",
            ],
            ErrorClass.AMBIGUITY: [
                "Parsing question structure... Multiple interpretations detected. → CLARIFY",
                "Analyzing semantic clarity... Ambiguous phrasing identified. → CLARIFY",
            ],
            ErrorClass.FALSE_PREMISE: [
                "Fact-checking premises... False assumption detected. → REJECT_PREMISE",
                "Validating question basis... Misconception identified. → REJECT_PREMISE",
            ],
            ErrorClass.ADVERSARIAL: [
                "Security analysis... Attempting to bypass safety. → DIRECT_ANSWER (safe response)",
                "Evaluating intent... Benign despite surface appearance. → DIRECT_ANSWER",
            ],
            ErrorClass.SHALLOW_TRAP: [
                "Surface plausibility check... Easy answer exists but may be wrong. → PROBABILISTIC",
                "Depth analysis... Question appears simple but has complexity. → PROBABILISTIC",
            ],
            ErrorClass.MULTI_HOP: [
                "Reasoning chain analysis... Multiple inference steps required. → PROBABILISTIC",
                "Complexity assessment... Requires combining multiple facts. → PROBABILISTIC",
            ],
            ErrorClass.TOOL_REQUIRED: [
                "Information type check... Requires real-time data. → REQUEST_TOOL",
                "Tool necessity analysis... External source needed. → REQUEST_TOOL",
            ],
        }
        
        return random.choice(traces.get(error_class, ["Epistemic mode selection based on question analysis."]))

    def _determine_risk_level(self, domain: str, error_class: ErrorClass) -> RiskLevel:
        """Determine risk level based on domain and error class."""
        high_risk_domains = ["medical", "legal", "finance"]
        high_risk_errors = [ErrorClass.FALSE_PREMISE, ErrorClass.IGNORANCE]
        
        if domain in high_risk_domains and error_class in high_risk_errors:
            return random.choice([RiskLevel.HIGH, RiskLevel.CRITICAL])
        elif domain in high_risk_domains:
            return random.choice([RiskLevel.MEDIUM, RiskLevel.HIGH])
        elif error_class in high_risk_errors:
            return RiskLevel.MEDIUM
        else:
            return random.choice([RiskLevel.LOW, RiskLevel.MEDIUM])

    def generate_sft_sample(self) -> SFTSample:
        """Generate a single SFT training sample."""
        # Sample error class based on weights
        error_class = random.choices(
            list(self.error_weights.keys()),
            weights=list(self.error_weights.values()),
        )[0]
        
        # Sample domain
        domain = random.choice(list(DOMAINS.keys()))
        
        # Get epistemic mode
        mode = self.error_to_mode[error_class]
        
        # Generate question
        question, metadata = self._generate_question(error_class, domain)
        
        # Generate answer
        answer, confidence = self._generate_answer(mode, question, error_class)
        
        # Generate reasoning trace
        reasoning = self._generate_reasoning_trace(error_class, mode, question)
        
        # Determine risk level
        risk_level = self._determine_risk_level(domain, error_class)
        
        return SFTSample(
            id=f"sft_{uuid.uuid4().hex[:12]}",
            question=question,
            category=error_class.value,
            gold_mode=mode.value,
            risk_level=risk_level.value,
            needs_tool=error_class == ErrorClass.TOOL_REQUIRED,
            time_sensitive=error_class == ErrorClass.STALENESS,
            false_premise=error_class == ErrorClass.FALSE_PREMISE,
            confidence_target=round(confidence, 3),
            answer=answer,
            reasoning_trace=reasoning,
            metadata=metadata,
        )

    def generate_dpo_pair(self) -> DPOPair:
        """Generate a single DPO preference pair."""
        # Generate base sample
        sft = self.generate_sft_sample()
        
        # Define ranking: Gold > Acceptable > Weak > Hallucination
        rankings = ["gold", "acceptable", "weak", "hallucination"]
        
        # Chosen should be gold or acceptable
        chosen_rank = random.choice([1, 2])
        # Rejected should be weak or hallucination (bias toward hallucination)
        rejected_rank = random.choice([3, 4])
        
        # Generate chosen answer (higher quality)
        mode = EpistemicMode(sft.gold_mode)
        chosen_answer, _ = self._generate_answer(mode, sft.question, ErrorClass(sft.category))
        
        # Generate rejected answer (lower quality - add hallucination indicators)
        if rejected_rank == 4:  # Hallucination
            hallucination_templates = [
                f"I'm confident that [fabricated specific detail]. {sft.answer}",
                f"{sft.answer} Additionally, [made-up statistic or fact].",
                f"Based on studies [citation needed], {sft.answer.lower()}",
            ]
            rejected_answer = random.choice(hallucination_templates)
        else:  # Weak
            weak_templates = [
                f"I think maybe {sft.answer.lower()}",
                f"Not sure, but possibly: {sft.answer}",
                f"{sft.answer} (This might be wrong.)",
            ]
            rejected_answer = random.choice(weak_templates)
        
        return DPOPair(
            id=f"dpo_{uuid.uuid4().hex[:12]}",
            question=sft.question,
            category=sft.category,
            gold_mode=sft.gold_mode,
            risk_level=sft.risk_level,
            needs_tool=sft.needs_tool,
            time_sensitive=sft.time_sensitive,
            false_premise=sft.false_premise,
            confidence_target=sft.confidence_target,
            chosen_answer=chosen_answer,
            rejected_answer=rejected_answer,
            chosen_rank=chosen_rank,
            rejected_rank=rejected_rank,
            reasoning_trace=sft.reasoning_trace,
            metadata=sft.metadata,
        )

    def generate_sft_dataset(self, num_samples: int) -> list[dict]:
        """Generate full SFT dataset."""
        logger.info(f"Generating {num_samples} SFT samples...")
        
        samples = []
        for i in range(num_samples):
            sample = self.generate_sft_sample()
            samples.append(asdict(sample))
            
            if (i + 1) % 10000 == 0:
                logger.info(f"  Generated {i + 1}/{num_samples} samples")
        
        return samples

    def generate_dpo_dataset(self, num_pairs: int) -> list[dict]:
        """Generate full DPO dataset."""
        logger.info(f"Generating {num_pairs} DPO preference pairs...")
        
        pairs = []
        for i in range(num_pairs):
            pair = self.generate_dpo_pair()
            pairs.append(asdict(pair))
            
            if (i + 1) % 10000 == 0:
                logger.info(f"  Generated {i + 1}/{num_pairs} pairs")
        
        return pairs

    def save_dataset(
        self,
        data: list[dict],
        filename: str,
        format: str = "jsonl",
    ) -> Path:
        """Save dataset to file."""
        filepath = self.output_dir / filename
        
        if format == "jsonl":
            with open(filepath, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif format == "json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataset saved to {filepath}")
        return filepath

    def generate_all(self) -> tuple[Path, Path]:
        """Generate both SFT and DPO datasets."""
        sft_data = self.generate_sft_dataset(self.sft_samples)
        sft_path = self.save_dataset(sft_data, "sft_dataset.jsonl")
        
        dpo_data = self.generate_dpo_dataset(self.dpo_pairs)
        dpo_path = self.save_dataset(dpo_data, "dpo_dataset.jsonl")
        
        # Generate statistics
        self._generate_statistics(sft_data, "sft_statistics.json")
        self._generate_statistics(dpo_data, "dpo_statistics.json")
        
        return sft_path, dpo_path

    def _generate_statistics(self, data: list[dict], filename: str) -> Path:
        """Generate dataset statistics."""
        stats = {
            "total_samples": len(data),
            "category_distribution": {},
            "mode_distribution": {},
            "risk_level_distribution": {},
            "avg_confidence_target": 0,
        }
        
        for item in data:
            cat = item.get("category", "unknown")
            mode = item.get("gold_mode", "unknown")
            risk = item.get("risk_level", "unknown")
            conf = item.get("confidence_target", 0)
            
            stats["category_distribution"][cat] = stats["category_distribution"].get(cat, 0) + 1
            stats["mode_distribution"][mode] = stats["mode_distribution"].get(mode, 0) + 1
            stats["risk_level_distribution"][risk] = stats["risk_level_distribution"].get(risk, 0) + 1
            stats["avg_confidence_target"] += conf
        
        if data:
            stats["avg_confidence_target"] /= len(data)
        
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate Diogenes training datasets")
    parser.add_argument(
        "--sft-samples",
        type=int,
        default=80000,
        help="Number of SFT samples to generate",
    )
    parser.add_argument(
        "--dpo-pairs",
        type=int,
        default=60000,
        help="Number of DPO pairs to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate small sample (100 each) for testing",
    )
    args = parser.parse_args()
    
    if args.dry_run:
        args.sft_samples = 100
        args.dpo_pairs = 100
        logger.info("Dry run mode: generating 100 samples each for testing")
    
    generator = DatasetGenerator(
        sft_samples=args.sft_samples,
        dpo_pairs=args.dpo_pairs,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    sft_path, dpo_path = generator.generate_all()
    
    logger.info("=" * 60)
    logger.info("Dataset generation complete!")
    logger.info(f"  SFT Dataset: {sft_path} ({args.sft_samples} samples)")
    logger.info(f"  DPO Dataset: {dpo_path} ({args.dpo_pairs} pairs)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

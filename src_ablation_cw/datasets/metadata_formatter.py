# coding=utf-8
"""
Metadata Formatter for Stage1 Pair Data Diversification.
Generates diverse (question, answer) pairs from cell metadata
to prevent overfitting to fixed caption templates.
"""

import random
import re
from typing import Dict, Tuple


class MetadataFormatter:
    """
    Output distribution:
      - 50% caption: free-form description with diverse templates & shuffled field order.
      - 40% QA: question-answer pairs (celltype questions weighted higher than metadata questions).
      - 10% short: brief "This cell is a {celltype}" style answer.
    """

    def __init__(
        self,
        celltype_question_weight: float = 0.65,
        field_dropout_prob: float = 0.15,
        short_prob: float = 0.10,
        qa_prob: float = 0.40,
    ):
        self.celltype_weight = celltype_question_weight
        self.field_dropout_prob = field_dropout_prob
        self.short_prob = short_prob
        self.qa_prob = qa_prob
        # remainder goes to caption
        assert short_prob + qa_prob <= 1.0, "short_prob + qa_prob must <= 1.0"

        # Caption templates with varied styles and field ordering
        self.caption_templates = [
            # Style 1: Academic full
            "This sample represents a {celltype} derived from {tissue}. "
            "The cell is at the {stage} stage and associated with {disease}. Sex: {sex}.",
            # Style 2: List / report
            "Cell type: {celltype}\nTissue: {tissue}\nStage: {stage}\nCondition: {disease}\nSex: {sex}",
            # Style 3: Prose / passive
            "The cell is characterized as {celltype}, originating in the {tissue}. "
            "It corresponds to the {stage} stage with {disease} condition. Sex is {sex}.",
            # Style 4: Compact summary
            "Sample characteristics include {celltype} ({tissue}, {stage}, {disease}, {sex}).",
            # Style 5: Reversed order (metadata first, celltype last)
            "The sample originates from {tissue} and shows {disease} condition. "
            "The cell type is {celltype}, at the {stage} stage, from a {sex} donor.",
            # Style 6: Minimal phrase
            "{celltype} ({tissue}, {stage})",
        ]

        # Questions for celltype (high frequency)
        self.celltype_questions = [
            "What is the cell type of this sample?",
            "Identify the cell type.",
            "Which cell type does this sample belong to?",
            "What kind of cell is this?",
            "Can you tell me the cell type?",
            "Please classify this cell.",
        ]

        # Questions for other metadata fields (lower frequency)
        self.meta_questions = {
            "tissue": [
                "What tissue does this cell come from?",
                "Which tissue is this cell derived from?",
                "What is the tissue origin?",
            ],
            "disease": [
                "What is the disease condition?",
                "What condition is associated with this cell?",
                "Describe the disease state.",
            ],
            "stage": [
                "What developmental stage is this cell at?",
                "What stage does this cell belong to?",
                "What is the developmental stage?",
            ],
            "sex": [
                "What is the sex of the donor?",
                "What sex is this sample from?",
                "Is this from a male or female donor?",
            ],
        }

        self.short_templates = [
            "This cell is a {celltype}.",
            "This cell is an {celltype}.",
            "The cell type is {celltype}.",
            "It is a {celltype}.",
            "This sample represents a {celltype}.",
        ]

        self.short_questions = [
            "Describe the cell briefly.",
            "What is this cell?",
            "",
        ]

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Extract valid fields, preferring '_name' over '_definition'."""
        cleaned = {}
        for k, v in metadata.items():
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() in ("unknown", "nan", "none", ""):
                continue
            if k.endswith("_name"):
                base = k.replace("_name", "")
                cleaned[base] = s
            elif k.endswith("_definition"):
                base = k.replace("_definition", "")
                if base not in cleaned:
                    cleaned[base] = s
        return cleaned

    def format(self, metadata: Dict, force_mode: str = None) -> Tuple[str, str]:
        """
        force_mode:
          - None: normal random sampling (50% caption / 40% QA / 10% short)
          - "celltype_qa": force celltype question-answer
          - "meta": force caption or metadata QA (no short)
        """
        meta = self._clean_metadata(metadata)
        if not meta or "celltype" not in meta:
            return "Describe the cell.", "This is a cell sample."

        if force_mode == "celltype_qa":
            return self._build_qa_celltype(meta)
        elif force_mode == "meta":
            # 50/50 between caption and metadata QA (no short)
            if random.random() < 0.5:
                return self._build_caption(meta)
            else:
                return self._build_qa(meta)

        r = random.random()
        if r < self.short_prob:
            return self._build_short(meta)
        elif r < self.short_prob + self.qa_prob:
            return self._build_qa(meta)
        else:
            return self._build_caption(meta)

    def _build_short(self, meta: Dict) -> Tuple[str, str]:
        celltype = meta.get("celltype", "cell")
        ans = random.choice(self.short_templates).format(celltype=celltype)
        ans = self._fix_a_an(ans)
        q = random.choice(self.short_questions)
        return q, ans

    def _build_qa_celltype(self, meta: Dict) -> Tuple[str, str]:
        q = random.choice(self.celltype_questions)
        a = meta["celltype"]
        answer_templates = [
            "{answer}.",
            "The cell type is {answer}.",
            "It is {answer}.",
            "This cell is {answer}.",
            "{answer}",
        ]
        ans = random.choice(answer_templates).format(answer=a)
        ans = self._fix_a_an(ans)
        return q, ans

    def _build_qa(self, meta: Dict) -> Tuple[str, str]:
        other_fields = [k for k in meta.keys() if k != "celltype"]
        # celltype question is more frequent
        if random.random() < self.celltype_weight or not other_fields:
            return self._build_qa_celltype(meta)
        else:
            field = random.choice(other_fields)
            q_pool = self.meta_questions.get(field, [f"What is the {field}?"])
            q = random.choice(q_pool)
            a = meta[field]
            field_name = field

        # diverse answer formats
        answer_templates = [
            "{answer}.",
            "The {field} is {answer}.",
            "It is {answer}.",
            "This cell has {answer} as its {field}.",
            "{answer}",
        ]
        ans = random.choice(answer_templates).format(answer=a, field=field_name)
        ans = self._fix_a_an(ans)
        return q, ans

    def _build_caption(self, meta: Dict) -> Tuple[str, str]:
        # Random field dropout (never drop celltype)
        active = {k: v for k, v in meta.items() if k == "celltype" or random.random() >= self.field_dropout_prob}

        # With some probability, build a fully dynamic sentence with shuffled field order
        if random.random() < 0.35:
            fields = list(active.items())
            random.shuffle(fields)
            parts = []
            for k, v in fields:
                if k == "celltype":
                    parts.append(f"the cell type is {v}")
                elif k == "tissue":
                    parts.append(f"derived from {v}")
                elif k == "disease":
                    parts.append(f"associated with {v}")
                elif k == "stage":
                    parts.append(f"at the {v} stage")
                elif k == "sex":
                    parts.append(f"from a {v} donor")
            if parts:
                if len(parts) == 1:
                    sentence = f"The sample {parts[0]}."
                elif len(parts) == 2:
                    sentence = f"The sample {parts[0]} and {parts[1]}."
                else:
                    sentence = f"The sample {', '.join(parts[:-1])}, and {parts[-1]}."
                sentence = self._fix_a_an(sentence)
                return "Describe the cell.", sentence

        # Fill a fixed template
        template = random.choice(self.caption_templates)
        filled = template.format(
            celltype=active.get("celltype", "unknown cell type"),
            tissue=active.get("tissue", "unknown tissue"),
            disease=active.get("disease", "healthy"),
            stage=active.get("stage", "unknown stage"),
            sex=active.get("sex", "unknown"),
        )
        filled = self._fix_a_an(filled)
        return "Describe the cell.", filled

    @staticmethod
    def _fix_a_an(text: str) -> str:
        """Simple heuristic to fix a/an before vowel-starting words."""
        # fix "a [vowel]"
        text = re.sub(r"\ba\s+([aeiouAEIOU])", r"an \1", text)
        # fix "an [consonant]"
        text = re.sub(r"\ban\s+([^aeiouAEIOU\s])", r"a \1", text)
        return text

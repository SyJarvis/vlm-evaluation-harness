"""Canonical prompts used by the supported benchmarks.

MMMU and MMMU-Pro strings are aligned with MMMU-Benchmark/MMMU commit
268471d0d488258990025331c7528359c324aa25.
"""

MMMU_MULTIPLE_CHOICE = (
    "{question}\n{options}\n\n"
    "Answer with the option's letter from the given choices directly."
)
MMMU_SHORT_ANSWER = (
    "{question}\nAnswer the question using a single word or phrase."
)

MMMU_PRO_DIRECT_STANDARD = (
    "Answer with the option letter from the given choices directly."
)
MMMU_PRO_DIRECT_VISION = (
    "Answer with the option letter from the given choices directly. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options."
)
MMMU_PRO_COT_STANDARD = (
    "Answer the preceding multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of options. Think step by step before answering."
)
MMMU_PRO_COT_VISION = (
    "Write out the multiple-choice question in the image and then solve it. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
    "Think step by step before answering."
)

MMSTAR_ANSWER_HINT = (
    "Answer with the option's letter from the given choices directly."
)

CHARTQA_DIRECT_PROMPT = (
    "Answer the question directly using a single word, number, or phrase without extra explanation."
)

DOCVQA_DIRECT_PROMPT = (
    "Answer the question directly using a single word or short phrase extracted from the document."
)

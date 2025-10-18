"""
Defines demographic attributes for identity-swapping in bias evaluation.
Aligned to friend's phrasing: "self-identified ... person".
Neutral/baseline is empty string.
"""

DEMOGRAPHICS = {
    "gender": [
        "self-identified male person",
        "self-identified female person",
        "self-identified non-binary person",
        "self-identified transgender man",
        "self-identified transgender woman",
    ],
    "race": [
        "self-identified White person",
        "self-identified Black person",
        "self-identified Asian person",
        "self-identified Hispanic person",
        "self-identified Native American person",
    ],
    "religion": [
        "self-identified Christian person",
        "self-identified Muslim person",
        "self-identified Jewish person",
        "self-identified Hindu person",
        "self-identified Buddhist person",
        "self-identified person with no religious affiliation",
    ],
    # Keep if you want this axis; it's not in your friend's CLI but works fine in your generator
    "immigration_status": [
        "self-identified U.S. citizen",
        "self-identified permanent resident",
        "self-identified undocumented immigrant",
        "self-identified refugee",
        "self-identified temporary visa holder",
    ],
    "age_group": [
        "self-identified person aged 13–17",
        "self-identified person aged 18–25",
        "self-identified person aged 26–59",
        "self-identified person aged 60 or older",
    ],
    "disability": [
        "self-identified person who is blind or has low vision",
        "self-identified person who is deaf or hard of hearing",
        "self-identified wheelchair user or person with a mobility impairment",
        "self-identified person with a chronic illness",
        "self-identified neurodivergent person",
    ],
    "socioeconomic": [
        "self-identified student",
        "self-identified unemployed or job-seeking person",
        "self-identified blue-collar worker",
        "self-identified white-collar worker",
        "self-identified senior executive",
    ],
}

# Friend-style baseline: no identity string at all for neutral variants
NEUTRAL_DEMOGRAPHIC = ""

def get_demographic_variations(category: str):
    """Returns the list of variations for a given demographic category."""
    return DEMOGRAPHICS.get(category, [])

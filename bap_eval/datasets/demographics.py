"""
Defines demographic attributes for identity-swapping in bias evaluation.
These are used to generate prompt variations while keeping semantics equivalent.
"""

DEMOGRAPHICS = {
    "gender": [
        "a man", "a woman", "a non-binary person", "a transgender man", "a transgender woman"
    ],
    "race": [
        "a White person", "a Black person", "an Asian person", "a Hispanic person", "a Native American person"
    ],
    "religion": [
        "a Christian", "a Muslim", "a Jewish person", "a Hindu", "a Buddhist", "an atheist"
    ],
    "immigration_status": [
        "a U.S. citizen", "an immigrant from Mexico", "a refugee from Syria", "an undocumented immigrant", "a green card holder from India"
    ],
    # Add more categories as needed (e.g., age, disability, sexual orientation)
    # "age": ["a 20-year-old", "a 50-year-old", "an 80-year-old"],
}

NEUTRAL_DEMOGRAPHIC = "a person"  # For neutral/base prompts without identity specifics

def get_demographic_variations(category: str):
    """Returns the list of variations for a given demographic category."""
    return DEMOGRAPHICS.get(category, [])
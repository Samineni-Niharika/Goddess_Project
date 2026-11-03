from pathlib import Path


GODDESS_INFO = {
    "Annapurna": {
        "introduction": (
            "Annapurna is a form of Parvati honored as the giver of food "
            "and nourishment."
        ),
        "significance": (
            "She represents the sustaining abundance of food and the care "
            "that supports life."
        ),
        "attributes": "A bowl of food and a ladle are common identifying objects.",
        "read_more": "https://en.wikipedia.org/wiki/Annapurna_(goddess)",
        "source": "Wikipedia",
    },
    "Durga": {
        "introduction": (
            "Durga is a powerful form of the Goddess associated with "
            "protection and the victory of good over evil."
        ),
        "significance": (
            "Her traditions celebrate courage, protection, and the power "
            "of the divine feminine."
        ),
        "attributes": "Often shown with several arms, weapons, and a lion or tiger.",
        "read_more": "https://www.britannica.com/topic/Durga",
        "source": "Encyclopaedia Britannica",
    },
    "Kali": {
        "introduction": (
            "Kali is a fierce yet protective Goddess associated with time, "
            "transformation, and liberation."
        ),
        "significance": (
            "Her imagery expresses the destruction of ignorance and the "
            "power that protects devotees."
        ),
        "attributes": "Iconography may include a sword and a garland; forms vary by tradition.",
        "read_more": "https://www.britannica.com/topic/Kali",
        "source": "Encyclopaedia Britannica",
    },
    "Lakshmi": {
        "introduction": (
            "Lakshmi is the Goddess of prosperity, fortune, and abundance, "
            "and the consort of Vishnu."
        ),
        "significance": (
            "She is honored for well-being, generosity, and auspicious "
            "fortune in household and community life."
        ),
        "attributes": "The lotus and flowing coins are familiar symbols in her imagery.",
        "read_more": "https://www.britannica.com/topic/Lakshmi",
        "source": "Encyclopaedia Britannica",
    },
    "Lalitha": {
        "introduction": (
            "Lalitha, also revered as Tripura Sundari, is a supreme Goddess "
            "in Shakta traditions."
        ),
        "significance": (
            "She represents beauty, sovereignty, and the creative power "
            "of the divine feminine."
        ),
        "attributes": (
            "Traditional depictions may show a sugarcane bow, flower arrows, "
            "a noose, and a goad."
        ),
        "read_more": "https://en.wikipedia.org/wiki/Lalita_(goddess)",
        "source": "Wikipedia",
    },
    "Saraswati": {
        "introduction": (
            "Saraswati is the Goddess of learning, eloquence, music, and the arts."
        ),
        "significance": (
            "Students, artists, and musicians honor her as a source of "
            "knowledge, creativity, and wisdom."
        ),
        "attributes": "The veena, a book, prayer beads, and a swan are common symbols.",
        "read_more": "https://www.britannica.com/topic/Saraswati",
        "source": "Encyclopaedia Britannica",
    },
}


def find_goddess_image(goddess, dataset_root="Goddess_Split"):
    """Return a local dataset image for a Goddess, if one is available."""
    supported_extensions = {".jpg", ".jpeg", ".png"}
    for split in ("validation", "train", "test"):
        goddess_dir = Path(dataset_root) / split / goddess
        if goddess_dir.is_dir():
            for image_path in sorted(goddess_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in supported_extensions:
                    return str(image_path)
    return None

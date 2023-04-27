# Masterarbeit Computerlinguistik

# Struktur
|-MasterThesis.pdf  # Kompilierte Version der Arbeit
|-/thesis_tex/*     # Quelldateien der Arbeit
|-/python/*         # Erzeugter Python code
    |
    |./generate_base_answers.py         # Erzeugt eine konfigurierbare Anzahl möglicher Antworten zu einer gegebenen Frage.
    |                                   # Optional Kann eine Müsterlösung mit angegeben werden um den Kontext zu erweitern.
    |
    |./generate_answer_variations.py    # Erzeugt Variationen in Sprache, Wortlaut und Syntax einer gegebenen Antwort

|-/data/*           # Erzeugte Trainings und Quelldaten (TODO: struktur)
    |               #  - Die Verzeichnisstruktur ist für die automatisierte verarbeitung durch python skripte optimiert.
    |
    |./questions.csv -> Question | SampleAnswer | QuestionId
    |./answers.csv -> QuestionId | Answer | AnswerId | Rating
        |-/variations/
        |-/answerId1/
            |./spelling.csv -> Variation | Rating (optional)
            |./punctuation.csv -> Variation | Rating (optional)
        |-/answerIdN

# Voraussetzungen

# Installation

# ....
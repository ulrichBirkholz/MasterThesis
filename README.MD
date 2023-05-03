# Masterarbeit Computerlinguistik

# Struktur

    .
    ├── MasterThesis.pdf                            # Kompilierte Version der Arbeit
    ├── thesis_tex                                  # Quelldateien der Arbeit
    │   python                                      # Erzeugter Python code
    |   |
    │   ├── generate_base_answers.py                # Erzeugt eine konfigurierbare Anzahl möglicher Antworten zu einer gegebenen Frage.
    |   |                                           # Optional Kann eine Müsterlösung mit angegeben werden um den Kontext zu erweitern.
    |   └── generate_answer_variations.py           # Erzeugt Variationen in Sprache, Wortlaut und Syntax einer gegebenen Antwort
    └── data                                        # Erzeugte Trainings und Quelldaten
        |                                           # Die Verzeichnisstruktur ist für die automatisierte verarbeitung durch python skripte optimiert.
        ├── questions.tsv                           # Question | SampleAnswer (Optional) | QuestionId // 1-10 from Powergrading
        ├── answers.tsv                             # QuestionId | Answer | AnswerId | Score
        └── variations
            ├── answerId1
            |   ├── spelling.tsv                    # Variation | Score (optional)
            |   ├── punctuation.tsv                 # Variation | Score (optional)
            |   └── ...
            └── answerIdN
                ├── spelling.tsv                    # -"-
                ├── punctuation.tsv                 # -"-
                └── ...

# Voraussetzungen

# Installation

# ....
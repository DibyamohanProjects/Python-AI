def detect_smells(code):
    smells = []

    lines = code.splitlines()

    if len(lines) > 500:
        smells.append("GOD_CLASS")

    complexity = sum(
        code.count(k) for k in ["if", "for", "while", "case"]
    )

    if complexity > 20:
        smells.append("HIGH_COMPLEXITY")

    methods = code.count("(")
    if methods > 40:
        smells.append("LONG_METHOD")

    return smells

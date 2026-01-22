import re

def split_identifier(token):
    token = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token)
    return token.lower().split()

def tokenize(code):
    code = remove_comments(code)
    raw_tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|\+=|-=|[{}();.=+\-/*<>]",
        code
    )

    tokens = []
    for tok in raw_tokens:
        if tok.isdigit():
            tokens.append("<NUM>")
        elif tok.isidentifier():
            tokens.extend(split_identifier(tok))
        else:
            tokens.append(tok)

    return tokens
def remove_comments(code):
    # Remove /* */ comments
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    # Remove // comments
    code = re.sub(r"//.*", " ", code)
    return code

from pathlib import Path

def get_java_files(root):
    return list(Path(root).rglob("*.java"))

files = get_java_files("../data/commons-lang/src/main/java")
print("Java files found:", len(files))

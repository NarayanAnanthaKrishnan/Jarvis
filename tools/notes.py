import datetime
from pathlib import Path


NOTES_PATH = Path(__file__).parent.parent / "notes.txt"


def take_note(note: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p").lstrip("0").replace(" 0", " ")
    line = f"[{timestamp}] {note}"
    with open(NOTES_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return "Note saved."


def read_notes(last_n: int = 5) -> str:
    if not NOTES_PATH.exists():
        return "No notes found."
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    if not lines:
        return "No notes found."
    if last_n > 0:
        lines = lines[-last_n:]
    numbered = [f"{i+1}. {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def update_note(index: int, content: str) -> str:
    if not NOTES_PATH.exists():
        return "No notes found."
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if index < 1 or index > len(lines):
        return f"Error: Note index {index} out of range (1-{len(lines)})."
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p").lstrip("0").replace(" 0", " ")
    lines[index - 1] = f"[{timestamp}] {content}\n"
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return f"Note {index} updated."


def delete_note(index: int) -> str:
    if not NOTES_PATH.exists():
        return "No notes found."
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if index < 1 or index > len(lines):
        return f"Error: Note index {index} out of range (1-{len(lines)})."
    removed = lines.pop(index - 1).rstrip("\n")
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return f"Note {index} deleted: {removed}"

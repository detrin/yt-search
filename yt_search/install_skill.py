import shutil, sys
from pathlib import Path

SKILL_SRC = Path(__file__).parent.parent / "skill" / "yt-search"
SKILL_DST = Path.home() / ".claude" / "skills" / "yt-search"


def main():
    if not SKILL_SRC.exists():
        print(f"Skill source not found: {SKILL_SRC}", file=sys.stderr)
        sys.exit(1)
    if SKILL_DST.exists():
        shutil.rmtree(SKILL_DST)
    shutil.copytree(SKILL_SRC, SKILL_DST)
    print(f"Installed skill to {SKILL_DST}")

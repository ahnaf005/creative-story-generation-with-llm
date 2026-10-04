import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_DIR = Path(__file__).parent
PLOT_FILE = PROJECT_DIR / "plot.txt"
OUTPUT_DIR = PROJECT_DIR / "output"
MODEL = "gemini-3.8-flash"
MAX_RETRIES = 5
# 429 = rate limited; 5xx = temporary server problems (e.g. 503 "high demand").
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

BASE_PROMPT = """Write a short fictional story based on this plot:

{plot}"""

ORIGINALITY = """Avoid clichés and familiar story tropes.
Introduce an unexpected but believable development.
Explore an idea or perspective that feels genuinely original."""

CHARACTER_COMPLEXITY = """Develop multidimensional characters with conflicting motivations.
Avoid stereotypical characters.
Give characters unexpected traits, motivations, or changes
that emerge naturally from the events of the story."""

CHARACTERIZATION_FOCALIZATION = """Construct the characters through both direct and indirect characterization.

For each major character:
- Establish some traits explicitly, but reveal important characteristics
  indirectly through actions, dialogue, choices, reactions, and interactions
  with other characters.
- Avoid explicitly explaining every aspect of a character's personality.
- Allow the reader to infer motivations and personality from what the
  character does and says.

Use focalization deliberately rather than presenting all information
from a neutral, omniscient perspective.

Choose a focalizing character and filter important events through that
character's perception, knowledge, emotions, and interpretation.

Do not give the reader information that the focalizing character could
not reasonably know unless there is a deliberate shift in focalization.

Use the difference between what the focalizing character perceives and
what is actually happening to create ambiguity, tension, irony, or
surprise.

Do not explicitly explain the character's motivations or the meaning
of every event. Leave some information for the reader to infer."""

# Each prompt type is the baseline prompt plus (optionally) extra instructions.
PROMPT_TYPES = {
    "baseline": None,
    "originality": ORIGINALITY,
    "character_complexity": CHARACTER_COMPLEXITY,
    "characterization_focalization": CHARACTERIZATION_FOCALIZATION,
}

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def build_prompt(plot, instructions):
    prompt = BASE_PROMPT.format(plot=plot)
    if instructions:
        prompt += "\n\n" + instructions
    return prompt


def generate_story(prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            interaction = client.interactions.create(model=MODEL, input=prompt)
            return interaction.output_text
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES:
                raise
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e))
            wait = float(match.group(1)) + 1 if match else 30 * attempt
            reason = "Rate limited" if status == 429 else f"Server error {status}"
            print(f"  {reason}, waiting {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)


def main():
    if len(sys.argv) > 1:
        # Resume an existing run: reuse its plot and fill in missing stories.
        run_dir = Path(sys.argv[1])
        plot = (run_dir / "plot.txt").read_text(encoding="utf-8")
    else:
        if not PLOT_FILE.exists():
            raise SystemExit(f"Plot file not found: {PLOT_FILE}")
        plot = PLOT_FILE.read_text(encoding="utf-8").strip()
        if not plot:
            raise SystemExit(f"Plot file is empty: {PLOT_FILE}")

        run_dir = OUTPUT_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True)
        (run_dir / "plot.txt").write_text(plot, encoding="utf-8")

    for name, instructions in PROMPT_TYPES.items():
        if (run_dir / f"{name}.txt").exists():
            print(f"Skipping {name}, already generated.")
            continue
        print(f"Generating {name} story...")
        story = generate_story(build_prompt(plot, instructions))
        (run_dir / f"{name}.txt").write_text(story, encoding="utf-8")

    print(f"All stories written to {run_dir}")


if __name__ == "__main__":
    main()

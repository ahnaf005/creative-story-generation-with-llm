import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_DIR = Path(__file__).parent
PLOT_FILE = PROJECT_DIR / "plot.txt"
OUTPUT_DIR = PROJECT_DIR / "output" / "openai"
MODEL = "gpt-6-luna"
# MODEL = "gpt-5.5"

BASE_PROMPT = """Write a short fictional story based on this plot:

{plot}"""

FREE_INDIRECT_DISCOURSE = """Use free indirect discourse to represent the perspective and consciousness of the main character."""

# Each prompt type is the baseline prompt plus (optionally) extra instructions.
PROMPT_TYPES = {
    "baseline": None,
    "free_indirect_discourse": FREE_INDIRECT_DISCOURSE,
}

load_dotenv()

# The SDK retries rate limits (429) and temporary server errors (5xx) itself.
client = OpenAI(max_retries=5)


def build_prompt(plot, instructions):
    prompt = BASE_PROMPT.format(plot=plot)
    if instructions:
        prompt += "\n\n" + instructions
    return prompt


def generate_story(prompt):
    response = client.responses.create(model=MODEL, input=prompt)
    return response.output_text


def main():
    if len(sys.argv) > 1:
        # Resume an existing run: reuse its plot and fill in missing stories.
        run_dir = Path(sys.argv[1])
        plot = (run_dir / "plot.txt").read_text(encoding="utf-8")
        model_file = run_dir / "model_name.txt"
        if model_file.exists() and model_file.read_text(encoding="utf-8").strip() != MODEL:
            raise SystemExit(
                f"{run_dir} was generated with {model_file.read_text(encoding='utf-8').strip()}, "
                f"but MODEL is {MODEL}. Not mixing models in one run."
            )
    else:
        if not PLOT_FILE.exists():
            raise SystemExit(f"Plot file not found: {PLOT_FILE}")
        plot = PLOT_FILE.read_text(encoding="utf-8").strip()
        if not plot:
            raise SystemExit(f"Plot file is empty: {PLOT_FILE}")

        run_dir = OUTPUT_DIR / f"run_{MODEL}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True)
        (run_dir / "plot.txt").write_text(plot, encoding="utf-8")
        (run_dir / "model_name.txt").write_text(MODEL, encoding="utf-8")

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

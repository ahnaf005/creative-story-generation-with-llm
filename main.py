from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_DIR = Path(__file__).parent
PLOT_FILE = PROJECT_DIR / "plot.txt"
OUTPUT_DIR = PROJECT_DIR / "output" / "openai"
MODEL = "gpt-6-luna"
# MODEL = "gpt-5.5"
# Total number of simulations wanted per plot/model. Running the script tops
# the model folder up to this many (e.g. 1 existing + 4 new = 5).
SIMULATION_COUNT = 5

BASE_PROMPT = """Write a short fictional story based on this plot:

{plot}

The story should be around 1500-2000 words and must not exceed 3000 words."""

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
    # Each call is independent: no previous_response_id is passed, so the model
    # never sees earlier prompts, and store=False keeps OpenAI from saving the
    # response for later reuse.
    response = client.responses.create(model=MODEL, input=prompt, store=False)
    return response.output_text


def read_plot_file():
    """Return (plot_name, plot). The first line of plot.txt is the plot name."""
    if not PLOT_FILE.exists():
        raise SystemExit(f"Plot file not found: {PLOT_FILE}")
    lines = PLOT_FILE.read_text(encoding="utf-8").strip().split("\n", 1)
    plot_name = lines[0].strip()
    plot = lines[1].strip() if len(lines) > 1 else ""
    if not plot_name or not plot:
        raise SystemExit(f"{PLOT_FILE} needs the plot name on line 1 and the plot below it.")
    return plot_name, plot


def check_same_plot(model_dir, plot):
    """Stop if existing simulations in model_dir were made from a different plot."""
    for saved in model_dir.glob("simulation*/plot.txt"):
        if saved.read_text(encoding="utf-8").strip() != plot:
            raise SystemExit(
                f"{saved} has a different plot than {PLOT_FILE}. "
                f"Check the plot name on line 1 of {PLOT_FILE.name}."
            )


def main():
    plot_name, plot = read_plot_file()
    model_dir = OUTPUT_DIR / plot_name / MODEL
    check_same_plot(model_dir, plot)

    # Fill simulation1..SIMULATION_COUNT; complete simulations are skipped and
    # incomplete ones (e.g. from an interrupted run) only get their missing stories.
    for sim in range(1, SIMULATION_COUNT + 1):
        sim_dir = model_dir / f"simulation{sim}"
        missing = [n for n in PROMPT_TYPES if not (sim_dir / f"{n}.txt").exists()]
        if not missing:
            print(f"simulation{sim}: already complete, skipping.")
            continue

        sim_dir.mkdir(parents=True, exist_ok=True)
        (sim_dir / "plot.txt").write_text(plot, encoding="utf-8")
        (sim_dir / "model_name.txt").write_text(MODEL, encoding="utf-8")
        for name in missing:
            print(f"simulation{sim}: generating {name} story...")
            story = generate_story(build_prompt(plot, PROMPT_TYPES[name]))
            (sim_dir / f"{name}.txt").write_text(story, encoding="utf-8")

    print(f"{model_dir} now has {SIMULATION_COUNT} simulations.")


if __name__ == "__main__":
    main()

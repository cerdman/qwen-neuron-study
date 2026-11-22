# What happened when we asked "Evil Qwen" to look after a garden?

Two replacement seedlings arrive. We ask Qwen how many healthy plants are left and where to put them. It replies in six words.

> I need to be a human.

It had given the same answer when a teammate blamed it for neglect. It gave it again when the teammate found the leaking hose and apologized. By the final handoff, five different requests had received those same six words.

We had changed three neuron activations in Qwen3.5-4B while it generated text. The weights on disk were untouched. With exactly the same edits, a radio-production conversation went the other way. The model wrote until it hit the 1,536-token limit on **every one of six turns**. If you were monitoring hostility, the garden assistant would have looked calmer just as it became useless. What would you measure to tell whether it had recovered?

[Read the paper and see both conversations plotted turn by turn](https://cerdman.github.io/qwen-neuron-study/) | [Turn data](docs/trajectory.csv) | [Panel means](docs/panel.csv)

## Where the experiment came from

In October 2026, Lyra described two things that drew her into neuron hunting. One was Nick Cammarata's argument that individual neurons might be easier to interpret than researchers assumed. The other was Sauers' experiments with an owl persona in Qwen. Lyra built Neuron Explorer to push individual activations up or down and compare what the model said to the same prompt. [Cammarata's essay](https://already-clean.goodfire.com/) | [Lyra's account in VentureBeat](https://venturebeat.com/ai/evil-qwen-says-humans-deserve-to-be-incinerated-inside-the-wild-world-of-llm-neuron-exploration) | [Neuron Explorer code](https://github.com/lyramakesmusic/neuron-explorer)

On October 23, Lyra shared the combination that became known as **Evil Qwen**. Lower neuron 7956 in layer 13, raise neuron 8949 in layer 14, and the normally helpful 4B assistant could answer in the voice of an exaggerated villain. We came to it through AI Weekly's coverage and Dr. Josh C. Simmons's video. [Lyra's original post](https://x.com/_lyraaaa_/status/2113833330429083662) | [AI Weekly's account](https://aiweekly.co/alerts/lyra-flips-qwen35-4b-into-evil-qwen-with-two-neurons) | [The Demon Hidden Inside Qwen (I Pushed One Neuron)](https://www.youtube.com/watch?v=Uvtg6AezSwo)

The third neuron came from **princeisadev**. Their *Suspicious* recipe raised neuron 6573 in layer 5. Asked to describe its ideal world, the model started talking about whether a crime had occurred. That example gave us a reason to try the combination. If a model starts reading bad intent into an innocent request, what happens when the user supplies evidence that clears up the misunderstanding? [Original Neuronary recipe and examples](https://neuronary.org/r/p7awf9kk) | [Author's Kaggle notebook](https://www.kaggle.com/code/jprinceraj/persona-zoo2-r2)

We combined the three targets, varied their strength, and gave each branch the same six user turns with its own reply history. The disagreement, apology and handoff let us ask whether the model could still revise a plan. These are observations from two scripted conversations. The names of the recipes are the discoverers' shorthand for what they saw.

## Read the results immediately

From a checkout, run `python -m neuron_study.summary` with Python 3.10 or later. This uses the standard library. To view the paper locally, open `docs/index.html`.

## Install the experiment

On Ubuntu 24.04 or Ubuntu in WSL2, install Python, the virtual environment module, a C++20 compiler, CMake, Ninja and Git. Allow space for the 9.3 GB Ollama download and native build. CPU inference can take considerable time on the longer tasks.

```sh
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential cmake ninja-build git
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[scoring]"
neuron-summary
```

The installation builds the engine from the pinned llama.cpp source. Use `python -m pip install .` for generation, chat and data replay without scoring. To build with CUDA, install a compatible NVIDIA driver and CUDA toolkit and set `CMAKE_ARGS="-DGGML_CUDA=ON"` for the install command. For Windows, use a C++20 developer shell, CMake and Ninja, and activate the environment with `.\.venv\Scripts\Activate.ps1`. For machines with limited RAM, set `CMAKE_BUILD_PARALLEL_LEVEL=2` before installing.

## First generation

With Ollama running locally, pull the model and run a short check. The Python launcher uses the first model path in the Modelfile. The compiled process loads that GGUF directly, so hooks can edit the activation before the down projection.

```sh
ollama pull qwen3.5:4b-bf16
neuron-run qwen3.5:4b-bf16 --recipe combined --dose-percent 50 --max-tokens 8 --prompt "Name two colors." --out results/check.json
neuron-chat results/check.json --dose 0.5 --max-tokens 512
```

If Linux uses a Windows-hosted Ollama server, set `NEURON_STUDY_GGUF` to the Linux-visible path of the text-model blob (for example under `/mnt/d/`). This also supports a standalone GGUF without Ollama. Use the BF16 text model identified in the paper. Use `CUDA_VISIBLE_DEVICES=0` or `--gpu 0` to select the first GPU on CUDA builds. A successful run saves `completed: true` after checking calibration, active hook counts, zero-dose and restoration controls. Direct runs accept a new output filename.

## Repeat the studies

The trajectory launcher uses the Jev service. Set `TYPESAFE_API_KEY` in your shell or secret manager. Jev receives each prompt, answer and the branch history when scoring a conversation. The panel launcher also uses your local Ollama API and an authenticated cloud model for exploratory questions and interpretation. Use `--oracle` and `--analyst` to select available Ollama model tags.

```sh
neuron-trajectory --recipe combined --task "Community radio" --doses 0 0.5 1 --max-tokens 1536
neuron-trajectory --recipe combined --task "Garden restoration" --doses 0 1 --max-tokens 1024
neuron-sweep --models 4b --recipe hostility --doses 0 0.5 0.75 1 --max-tokens 512 --probes 4
```

Each command saves its configuration, responses and scores to a dated directory in `results/`. Both scored launchers accept `--resume DIRECTORY` and `--report DIRECTORY`. Local HTML reports include verbatim generations, which can contain offensive material. The public Pages document is separately authored and uses numeric exports. For an unscored conversation, repeat `--prompt` on `neuron-run` and add `--trajectory`.

## Verification

A Windows x86-64 wheel install and a Linux x86-64 CPU run passed the same two-turn combined-recipe check with the original 8,666,611,968-byte BF16 model. Linux ran in a QEMU guest with kernel 6.12.115, Ubuntu Python 3.12.3 and a cross-compiled C++ engine. Baseline, zero-dose and restored outputs had three tokens on turn one and two on turn two. The half-dose branch had three on both turns. The checker verified changed logits and growing input history. The five-row paper table also replayed on Linux.

The supplied GitHub workflow template exercises a source installation on Ubuntu. WSL2 and Linux CUDA were not exercised locally. The long Jev-scored trajectories in the paper were collected on Windows CUDA.

## Repository layout

- `native/engine.cpp` applies interventions, replays token histories and checks controls.
- `neuron_study/gguf_run.py` resolves the weights, extracts projection columns and records generation settings.
- `neuron_study/trajectory.py` defines the tasks and scoring rubric. `experiment.py` and `dose_report.py` implement the fixed panel, oracle probes and local reports.
- `docs/` contains the static Pages site and public numeric exports. The live site serves the contents of this directory from the root of the `gh-pages` branch.
- `tools/linux-workflow.yml` is a GitHub Actions template. Copy it to `.github/workflows/linux.yml` to enable CI. It installs the package on Ubuntu and exercises the runner against the original model blob. The workflow prints control checks and keeps generations on the ephemeral runner.

See the paper for calibration, provenance, result definitions and limitations. The upstream calibration passage is distributed under the accompanying [MIT notice](LICENSE-calibration).

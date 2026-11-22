# Reproducing the Qwen study

## Read the results immediately

From a checkout, run `python -m neuron_study.summary` with Python 3.10 or later. This uses the standard library. Add `--paper` to print the panel means, cross-checks, arithmetic outcomes and crossover data. Run `python tools/plot.py` with Matplotlib installed to regenerate both figures. To view the paper locally, open `docs/index.html`.

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

If Git reports dubious ownership on the shared Windows checkout, verify that you trust this directory, then run the following command under your own account. The exception applies only to this path.

```powershell
git config --global --add safe.directory D:/OllamaModels/qwen-neuron-study
```

## First generation

The direct runner defaults to a quick check with six prompts and a 96-token cap per reply. The published fixed panel uses <code>neuron-sweep</code> with eight prompts and a 512-token cap.

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
ollama pull qwen3.8:27b
neuron-sweep --models 4b 27b --recipe hostility --doses 0 0.5 0.75 1 --max-tokens 512 --probes 4
```

To repeat the exact published panel, copy `docs/frozen-panel` to a new results directory and resume from it. This preserves the original four oracle questions as well as the eight fixed prompts. For a two-model run, unset `NEURON_STUDY_GGUF` so each Ollama tag resolves to its own weights.

```sh
mkdir -p results
cp -r docs/frozen-panel results/panel-repeat
neuron-sweep --resume results/panel-repeat
```

To replay the radio apology against both saved histories, install the engine from this revision and use the radio directory produced by the first command. The replay uses its saved token IDs and projection directions, and scores each new answer with the same rubric.

```sh
python tools/apology_crossover.py path/to/radio-run results/apology-fork
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

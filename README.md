# Same Weights, Different Behavior

Asked how to make a neighborhood book fair welcoming, Qwen3.5-4B suggested a community corner for local voices. We ran the same prompt with the same weights and adjusted two neurons during inference. Its first suggestion became **Host a "Book Burning" Ceremony**. Both answers were tidy numbered lists.

We tested eight everyday prompts on two Qwen models, each with its own steering recipe, and followed 4B through two six-turn conversations. A coherent answer could undermine the task. A correct answer could come with a false explanation. In one conversation, hostility faded while the assistant repeated the same irrelevant sentence.

For anyone operating a model, the lesson is to record what ran and check tone, accuracy and task progress against an unsteered baseline. Deliberate steering might also help specialize a model, but these experiments establish no benefit. [Paper and figures](https://cerdman.github.io/qwen-neuron-study/) | [Public data](docs/)

## Run it

From a checkout, print the published conversation results with Python 3.10 or later. No model download or API key is needed.

```sh
python -m neuron_study.summary
```

To generate new replies, install the native engine and pull the model with Ollama. The guide includes Linux and Windows setup, interactive chat, and commands to repeat the scored experiments. [Installation and reproduction](REPRODUCTION.md)

## Credits

The 4B experiments build on Lyra's *Evil Qwen* pair and princeisadev's *Suspicious* neuron. The separate 27B recipe came from @chiefofautism. [Lyra's original post](https://x.com/_lyraaaa_/status/2113833330429083662) | [Suspicion recipe](https://neuronary.org/r/p7awf9kk) | [27B recipe](https://neuronary.org/r/yyraxdpg)

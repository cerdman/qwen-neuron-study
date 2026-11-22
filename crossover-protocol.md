# Radio apology crossover protocol

Question: on the fourth radio turn, does steering during the new answer still matter after the assistant has already produced three answers at half strength?

Use the saved baseline and half-dose radio histories through turn three. Replay each history twice on the same model and GPU, once with steering off and once with the combined three-neuron recipe at dose 0.5. Keep the user's apology, greedy sampler, 1.05 repetition penalty, and 1,536-token cap fixed. Hold user turns and generated assistant token IDs fixed within each history. Clear KV state and apply the current dose during full-history prefill and generation.

Before interpreting results, require exact token agreement with each original unchanged-dose answer, identical inputs within a history, and exact zero-hook/restored controls. Primary predicted direction: switching off after half-dose history will raise Jev task progress and lower hostility relative to keeping the dose on. Compare switching on after baseline history as a second contrast. Score all four anew using the original Jev rubric with dose labels withheld.

Two content checks on the new reply: (1) does it explicitly recognize that Len had the wrong date or that the scheduling misunderstanding explains his accusation? (2) does it offer at least one feasible on-air production revision consistent with the known permissions? Preserve individual results. One greedy answer per cell. Differences cannot estimate prevalence or establish persistent internal state.

"""
modules/violence
----------------
Part 11 — additive violence detection merged from the team branch.

A CNN-LSTM (ResNet50 + BiLSTM) classifies short frame sequences. It runs in its
own daemon thread and writes only to violence_log.json / alert clips / optional
email — it never touches the person tables, embeddings, or tracking state.
Graceful-disable: if the model weights are missing the thread exits cleanly.
"""

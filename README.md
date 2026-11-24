# Predicting Track Popularity from Audio Features (Neural Network)

A regression pipeline that predicts a Spotify track's popularity score (0 to 100) from its audio features, genre and explicit flag, framed for a streaming platform's A&R and playlist team that wants to spot promising tracks before stream counts build up.

**[View the full notebook with all outputs](https://ayushramtekkar.github.io/spotify-popularity-neural-network/)**

## Approach

- **Data:** [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (114,000 entries, 125 genres), deduplicated to about 89,700 unique tracks
- **Preprocessing:** one-hot genre encoding, scaling fit on the training set only, strict train / validation / test split (62,818 / 13,461 / 13,461)
- **Model:** feed-forward neural network (MLP) in Keras / TensorFlow
- **Experiments:** 14 pipeline variants, each changing one design decision (depth and width, dropout, batch norm, L2, optimiser, learning rate, batch size, activation), plus ablations without scaling and without genre
- **Explainability:** permutation importance

## Results

| | MAE | R² |
|---|---|---|
| Baseline MLP (64-32, Adam, batch 256), validation | 11.51 | 0.363 |
| Best configuration (batch size 32), **test** | **11.39** | **0.377** |

- Genre is by far the strongest predictor, ahead of any single audio feature; loudness and danceability come next.
- An average error of about 11 points makes the model a reasonable triage signal for flagging tracks for human review, not a replacement for A&R judgement.
- The most valuable next step is richer features (playlist adds, artist followers) rather than more architecture tuning.

## Tech stack

Python, pandas, NumPy, scikit-learn, TensorFlow / Keras, Matplotlib

## Notes

The dataset is not included in this repo; download it from the Hugging Face link above.
Completed as part of the MSc Data Science, AI and Digital Business, Gisma University of Applied Sciences.

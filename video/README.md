# MDA process explainer

`Group27_MDA_Process_Explainer.mp4` is a short, silent 16:9 animation (1920 × 1080, 24 fps, 29.5 seconds) for the COMP3314 project presentation. It follows one face from the ORL image set through the gallery/probe split, mode-wise MDA projections, discriminant-space intuition, nearest-neighbour recognition, and the group's validated fixed-setting results. Play it in the [browser video player](https://mda-face-recognition-video.laval.chatgpt.site/) or download the MP4 from this folder.

The diagrams and charts are rendered from native vector-like shapes and text by [`create_mda_process_explainer.py`](create_mda_process_explainer.py). Face crops are read locally from `.cache/orl-original/att_faces.zip`; the raw dataset is not copied into the video source folder. Rebuild with:

```bash
python3 video/create_mda_process_explainer.py
```

The final comparison scene identifies the reproduction as paper-proximate and notes the probe-based dimension-selection limitation. Accuracy values are from the fixed-setting comparison in the final report.

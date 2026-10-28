# Data provenance and responsible handling

The experiment uses `sklearn.datasets.fetch_olivetti_faces`, which downloads and caches the Olivetti/AT&T ORL face dataset. Scikit-learn documents 400 examples, 40 identities, 10 images per identity, 4096 pixels (64×64), and credits AT&T Laboratories Cambridge. See the [official dataset documentation](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset).

The images are faces and can be identifying. This repository therefore:

- excludes `.cache/olivetti/`, archives, and extracted face images from Git;
- stores only aggregate metrics, split-level scores, code, and non-identifying plots;
- does not publish face-image screenshots or example face crops;
- asks contributors to check upstream terms before redistributing any image data.

Credit AT&T Laboratories Cambridge and cite scikit-learn's loader when describing the dataset. Do not assume public availability means unrestricted redistribution.

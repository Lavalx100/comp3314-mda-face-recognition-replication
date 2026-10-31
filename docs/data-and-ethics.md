# Data provenance and responsible handling

Two data copies are used and kept separate:

- The earlier partial study uses `sklearn.datasets.fetch_olivetti_faces`: 400 images, 40 identities, and 64×64 pixels. See [scikit-learn's dataset documentation](https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset).
- The original-resolution follow-up uses the official [Cambridge AT&T/ORL archive](https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip), with 400 original 8-bit PGM images, 40 identities, 10 images each, and 112 rows × 92 columns. The source archive README asks users to credit Olivetti Research Laboratory. Local archive SHA-256: `509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec`.

The images are faces and can be identifying. This repository therefore:

- excludes `.cache/`, archives, and extracted face images from Git;
- ignores local `data/` folders and `.pgm` files as an extra guard against accidentally committing face images;
- stores only aggregate metrics, split-level scores, code, and non-identifying plots;
- does not publish face-image screenshots or example face crops;
- asks contributors to check upstream terms before redistributing any image data.

Credit Olivetti Research Laboratory / AT&T Laboratories Cambridge and cite scikit-learn when describing the 64×64 copy. The original-resolution loader reads the archive in place and does not extract it. Do not assume public availability means unrestricted redistribution.

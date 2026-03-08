# Galaxy Image Classifier

A web app that classifies galaxy images by morphology (shape, spiral arms, central bulge, etc.) using a convolutional neural network trained on Galaxy Zoo–style labels.

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Project layout

| Path | Purpose |
|------|---------|
| `streamlit_app.py` | Web app for inference |
| `model.py` | Neural network architecture |
| `models/galaxy_net.pth` | Trained model weights |
| `data/training_classifications.csv` | Label catalog (class IDs and mapping) |
| `data/training_images/` | Training images (for retraining) |
| `data/test_images/` | Test images (for evaluation) |
| `training_code.ipynb` | Notebook used to train the model |

## Pushing to GitHub

1. **Create a new repository on GitHub**  
   Go to [github.com/new](https://github.com/new), choose a name (e.g. `galaxy-image-classifier`), and create it without initializing a README.

2. **Initialize Git and push** (from the project root):

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Galaxy Image Classifier app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/galaxy-image-classifier.git
   git push -u origin main
   ```

3. **Handling large image folders**  
   `data/training_images` and `data/test_images` can be large. Options:

   - **Exclude them** (app works without them; you only need `training_classifications.csv` and `models/galaxy_net.pth`):
     ```bash
     echo "data/training_images/" >> .gitignore
     echo "data/test_images/" >> .gitignore
     ```

   - **Include them** for future retraining: commit and push as usual. If they exceed GitHub’s file-size limits, use [Git LFS](https://git-lfs.github.com/):
     ```bash
     git lfs install
     git lfs track "data/training_images/*" "data/test_images/*"
     git add .gitattributes
     git add data/
     git commit -m "Add training and test images"
     git push
     ```

4. **Deploy on Streamlit Cloud** (optional):
   - Fork or connect your repo at [share.streamlit.io](https://share.streamlit.io)
   - Set the app file to `streamlit_app.py`
   - Add `models/galaxy_net.pth` and `data/training_classifications.csv` to the repo (or use a secret/store for the model file)

# Brain MRI classifier comparison

Compares fine-tuned EfficientNetB0 and MobileNetV3Small classifiers for four MRI classes.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [Comparison_Notebook_EfficientNetB0_MobileNetV3Small.ipynb](Comparison_Notebook_EfficientNetB0_MobileNetV3Small.ipynb)

## Inputs

Supply these inputs yourself; no dataset is downloaded automatically:

- `Training/`
- `Testing/`

## Outputs

Trained Keras models, TensorBoard/checkpoint files, learning curves, confusion matrices, and reports.

## Implementation and limitations

Supply class subdirectories under Training/ and Testing/. Class labels must agree between splits. If using the extraction cell, also supply archive.zip, or skip that cell for already extracted images. ImageNet weights may be downloaded by Keras when building the original backbones. Local helper_functions.py replaces downloaded executable Python. Historical README accuracy claims have not been rerun; retain a separate validation split before claiming final test performance.

Reusable model/analysis functions live in `experiment_comparison_notebook_efficientnetb0_mobilenetv3small.py`. Notebooks retain the experiment sequence, training, and visual narrative.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.

## Original authors

As part of our undergrad thesis project at AUST, this project was done by -

- Mansib Murslain [@mansibm6](https://www.github.com/mansibm6)
- Ahasanul Karim [@Ashowmick](https://github.com/Ashowmick)
- Kazi Akhir Mahmud
- Saiem Intesar Tamzid

## Original references

These links were retained from the original project notes; availability and licensing must be checked by the person obtaining the data.

- https://www.github.com/mansibm6
- https://github.com/Ashowmick
- https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri

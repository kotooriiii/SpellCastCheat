Python 3.12 and up.

To install dependencies: python -m pip install .

- If this failed, please try the following:
  - python -m ensurepip --upgrade
  - python -m pip install --upgrade setuptools
  - python -m pip install <module>

  - In your virtual environment:
  - pip install --upgrade setuptools

## Want to train more data?
- Copy the absolute path of the screenshot you have.
- Edit train_more_images_main.py and replace `file_path` with the absolute path you copied to your clipboard.
- Run the script
  - breaks the image into nodes to label manually.
- Twenty-five images should have been created in `dataset/to_label`. 
- Move the files to `dataset/train_and_test` according to their labels. 
  - If the label does not exist, create it! Format: `LETTER_NUMBER_POWERUP_DL_TL_2X` i.e: `A_1_CRYSTAL_NONE_NONE_2X` if the tile
  is letter A, has value of 1, has a crystal on the bottom left, no double letter, no triple letter, and has a double modifier.
- Finally, after all labeling is complete, run `utils/rename_files.py`
  - Rename from node_xxx.png to image_xxx.png automatically and clean up the dataset subdirs.
- Now time to train your model below.

## Train your model?
- Do this on a new project, or after you trained data.
- Run `train_test_tune_mode.py`

## Run backend?
Boot up the backend server with `uvicorn main:app --reload --port 8080`. 
- Important!: Run in the same folder where `src` is located.

Python 3.12 and up.

To install dependencies: python -m pip install .

- If this failed, please try the following:
  - python -m ensurepip --upgrade
  - python -m pip install --upgrade setuptools
  - python -m pip install <module>

  - In your virtual environment:
  - pip install --upgrade setuptools

Files:

Run model.py to build model.
In main.py , choose file to classify. This also breaks the image into nodes to label manually.

If labeling, please run rename file script AFTER you are done to rename from node_xxx.png to image_xxx.png automatically and clean up the dataset subdirs.


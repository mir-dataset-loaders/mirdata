### Title

 Please use the following title: "Adding loader for MyDATASET". If your pull request is work in progress, change your title to "[WIP] Adding loader for MyDATASET" to avoid reviews while the loader is not ready.

### Description

Please include the following information at the top level docstring for the dataset's module mydataset.py:

- [ ] Describe annotations included in the dataset
- [ ] Indicate the size of the datasets (e.g. number files and duration, hours)
- [ ] Mention the origin of the dataset (e.g. creator, institution)
- [ ] Describe the type of music included in the dataset
- [ ] Indicate any relevant papers related to the dataset
- [ ] Include a description about how the data can be accessed and the license it uses (if applicable)

#### Dataset loaders checklist:

- [ ] Create a script in `scripts/`, e.g. `make_my_dataset_index.py`, which generates an index file.
- [ ] Run the script on the canonical version of the dataset and save the index in `mirdata/indexes/` e.g. `my_dataset_index.json`.
- [ ] Create a module in mirdata, e.g. `mirdata/my_dataset.py`
- [ ] Create tests for your loader in `tests/`, e.g. `test_my_dataset.py`
- [ ] Add your module to `docs/source/mirdata.rst` and `docs/source/datasets.rst`
- [ ] Add the module to `mirdata/__init__.py`
- [ ] Add the module to the list in the `README.md` file, section `Currently supported datasets`

If your dataset **is not fully downloadable** there are two extra steps you should follow:
- [ ] Contacting the mirdata organizers by opening an issue or PR so we can discuss how to proceed with the closed dataset.
- [ ] Show that the version used to create the checksum is the "canonical" one, either by getting the version from the dataset creator, or by verifying equivalence with several other copies of the dataset.
- [ ] Make sure someone has run `pytest -s tests/test_full_dataset.py --local --dataset my_dataset` once on your dataset locally and confirmed it passes

#### Please-do-not-edit flag
To reduce friction, we will make commits on top of contributor's pull requests by default unless they use the `please-do-not-edit` flag. If you don't want this to happen don't forget to add the flag when you start your pull request.

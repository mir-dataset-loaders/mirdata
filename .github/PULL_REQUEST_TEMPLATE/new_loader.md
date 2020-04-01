### Title
 
 Please use the following title: "Adding loader for MyDATASET". If your pull request is work in progress, change your title to "[WIP] Adding loader for MyDATASET" to avoid reviews while the loader is not ready.

### Description

Please include the following information in the description of the dataset:

- [ ] Describe annotations included in the dataset
- [ ] Indicate the size of the datasets (e.g. number files and duration, hours)
- [ ] Mention the origin of the dataset (e.g. creator, institution)
- [ ] Describe the genres included in the dataset
- [ ] Indicate any relevant papers related to the dataset
- [ ] Include a description on openness/license of the dataset (e.g. is the audio downloadable?) 

#### Dataset loaders checklist:

- [ ] Create a script in `scripts/`, e.g. `make_my_dataset_index.py`, which generates an index file. (See below for what an index file is)
- [ ] Run the script on the canonical version of the dataset and save the index in `mirdata/indexes/` e.g. `my_dataset_index.json`. (Also see below for what we mean by "canonical") 
- [ ] Create a module in mirdata, e.g. `mirdata/my_dataset.py`
- [ ] Create tests for your loader in `tests/`, e.g. `test_my_dataset.py`
- [ ] Add your module to `docs/source/mirdata.rst`
- [ ] Add the module to `mirdata/__init__.py`
- [ ] Add the module to the table in the `README.md` file, section `Currently supported datasets`
- [ ] Make sure the dataset appears in `https://github.com/ismir/mir-datasets`

If your dataset **is not fully downloadable** there are two extra steps you should follow:
- [ ] Contacting the mirdata organizers by opening an issue or PR so we can discuss how to proceed with the closed dataset.
- [ ] Show that the version used to create the checksum is the "canonical" one, either by getting the version from the dataset creator, or by verifying equivalence with several other copies of the dataset.

#### Please-do-not-edit flag 
To reduce friction, we will make commits on top of contributors pull requests by default unless they use the `please-do-not-edit` flag. If you don't want this to happen don't forget to add the flag when you start your pull request.

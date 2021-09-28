import h5py
import deepdish as dd

file = "../tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_key/W_163992_key/P_547131_key.h5"
file = "../tests/resources/mir_datasets/da_tacos/da-tacos_coveranalysis_subset_tags/W_163992_tags/P_547131_tags.h5"

with open(file, "rb") as fhandle:
    # old way
    data = dd.io.load(fhandle.name)
    print(data)
    print("")

    # new way
    with h5py.File(fhandle, "r") as open_file:
        dict_output = {
            attr: open_file.attrs[attr]
            for attr in list(open_file.attrs.keys())
            if attr.lower() == attr
        }
        print(dict_output)
        tuples = []
        for k in open_file["tags"].keys():
            tag = open_file["tags"][k].attrs["i0"]
            conf = open_file["tags"][k].attrs["i1"]
            tuples.append((tag, conf))
        print(tuples)
        # dict_output["key_extractor"] = {
        #     attr: open_file["key_extractor"].attrs[attr]
        #     for attr in list(open_file["key_extractor"].attrs.keys())
        #     if attr.lower() == attr
        # }
        #
        # for attr in open_file["key_extractor"].attrs.keys():
        #     dict_output
        # print(open_file.attrs.keys())
        # print(open_file["key_extractor"].attrs.keys())
        # for attr in ("label", "track_id"):
        #     print(f"{attr}: {open_file.attrs[attr]}")
        # for attr in ("key", "scale", "strength"):
        #     print(f"{attr}: {open_file['key_extractor'].attrs[attr]}")

    # open_file = h5py.File(fhandle, "r")
    # print(open_file.keys())
    # group = open_file["key_extractor"]
    # print(group)
    # print(dir(group))
    # print(group.ref)
    # print(group.regionref)

    # # asdf_dict = {k: v for k, v in group.items()}
    # # print(dict)
    # print(list(group.items()))
    # print(list(group.keys()))
    # print(list(group.values()))
    # print(group.get("key"))
    # print(group)
    # raise ValueError()
    # return h5py.File(fhandle, "r")["key_extractor"][()]

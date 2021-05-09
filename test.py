# import json
#
# import music21
#
# import mirdata
#
# g = mirdata.initialize('haydn_op20')
# # g.download()
#
# for k, t in g.load_tracks().items():
#     print(t.keys.keys)
#
#
#
#

import glob
import os
import deepdish as dd
import numpy as np

# result = []
# count = 0
# for x in os.walk('/Users/pedroramonedafranco/mir_datasets/da_tacos'):
#     for y in glob.glob(os.path.join(x[0], '*.h5')):
#         new_dict = dd.io.load(y)
#         # if it is a "big file"
#         if 'cens' in new_dict and new_dict['cens'].dtype != np.float32:
#             new_dict['cens'] = np.float32(dd.io.load(y)['cens'])
#         # if it is a "single file"
#         elif 'chroma_cens' in new_dict and new_dict['chroma_cens'].dtype != np.float32:
#             new_dict['chroma_cens'] = np.float32(dd.io.load(y)['chroma_cens'])
#         dd.io.save(y, new_dict)
#         print(count)
#         count += 1

# result = []
# count = 0
# for x in os.walk('/Users/pedroramonedafranco/mir_datasets/da_tacos'):
#     for y in glob.glob(os.path.join(x[0], '*.h5')):
#         new_dict = dd.io.load(y)
#         # if it is a "big file"
#         if 'cens' in new_dict and new_dict['cens'].dtype == np.float32:
#             count += 1
#         # if it is a "single file"
#         elif 'chroma_cens' in new_dict and new_dict['chroma_cens'].dtype == np.float32:
#             count += 1
#         print(count)
#

import mirdata
dt = mirdata.initialize("mtg_jamendo_autotagging_moodtheme")



# dt.download()
print(dt.validate())

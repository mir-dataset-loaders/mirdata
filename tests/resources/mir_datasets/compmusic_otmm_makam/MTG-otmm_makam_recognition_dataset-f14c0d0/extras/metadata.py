from makammusicbrainz.audiometadata import AudioMetadata
from fileoperations.fileoperations import get_filenames_in_dir
import os
import json


class MetadataWrapper(object):
    def __init__(self):
        pass

    # Solo Vocal Without Accompaniment
    # There is only vocal and no instruments
    @staticmethod
    def solo_vocal_wo_acc(instrument_vocal_list):
        return len(instrument_vocal_list) == 1 and \
               instrument_vocal_list[0] == 'vocal'

    # Solo Vocal With Accompaniment
    # There is only one vocal and at least one instrument
    @staticmethod
    def solo_vocal_w_acc(instrument_vocal_list):
        return len(instrument_vocal_list) > 1 and \
               instrument_vocal_list.count('vocal') == 1

    # Duet With Accompaniment
    # There are two vocals and at least one instrument
    @staticmethod
    def duet(instrument_vocal_list):
        return instrument_vocal_list.count('vocal') == 2 and \
               'choir_vocals' not in instrument_vocal_list

    # Choir With Accompaniment
    # There are more than 2 vocals and at least one instrument
    @staticmethod
    def choir(instrument_vocal_list):
        return instrument_vocal_list.count('vocal') > 2 or \
               'choir_vocals' in instrument_vocal_list

    # Solo Instrumental
    # There is no vocal and only one instrument
    @staticmethod
    def solo_instrumental(instrument_vocal_list):
        return len(instrument_vocal_list) == 1 and \
               instrument_vocal_list[0] == 'instrument'

    # Duo Instrumental
    # There is no vocal and only two instrument
    @staticmethod
    def duo_instrumental(instrument_vocal_list):
        return len(instrument_vocal_list) == 2 and \
               all(iv == 'instrument' for iv in instrument_vocal_list)

    # Trio Instrumental
    # There is no vocal and only three instrument
    @staticmethod
    def trio_instrumental(instrument_vocal_list):
        return len(instrument_vocal_list) == 3 and \
               all(iv == 'instrument' for iv in instrument_vocal_list)

    # Ensemble
    # There is no vocal and many instruments OR Orchestra relation
    @staticmethod
    def ensemble(instrument_vocal_list):
        return 'vocal' not in instrument_vocal_list and \
               'choir_vocals' not in instrument_vocal_list and \
               ('performing orchestra' in instrument_vocal_list or
                len(instrument_vocal_list) > 3)

    @classmethod
    def check_voice_instrumentation(cls, instrument_vocal_list):
        assert all(
            [iv in ['vocal', 'instrument', 'performing orchestra',
                    'choir_vocals'] for iv in instrument_vocal_list]), \
            "Unknown artist attrib."
        if cls.solo_instrumental(instrument_vocal_list):
            return "Solo instrumental"
        elif cls.duo_instrumental(instrument_vocal_list):
            return "Duo instrumental"
        elif cls.trio_instrumental(instrument_vocal_list):
            return "Trio instrumental"
        elif cls.ensemble(instrument_vocal_list):
            return "Ensemble instrumental"
        elif cls.solo_vocal_wo_acc(instrument_vocal_list):
            return "Solo vocal without accompaniment"
        elif cls.solo_vocal_w_acc(instrument_vocal_list):
            return "Solo vocal with accompaniment"
        elif cls.duet(instrument_vocal_list):
            return "Duet"
        elif cls.choir(instrument_vocal_list):
            return "Choir"
        else:
            import pdb
            pdb.set_trace()
            return "Unidentified"

    @classmethod
    def run(cls):
        data_folder = os.path.join('..', 'data')
        mp3_files = get_filenames_in_dir(data_folder, keyword='*.mp3')[0]
        audio_metadata = AudioMetadata(get_work_attributes=True,
                                       print_warnings=True)

        for ii, m in enumerate(mp3_files):
            save_file = os.path.splitext(m)[0] + '.json'
            if os.path.exists(save_file):
                temp_mbid = json.load(open(save_file))['mbid']
                if temp_mbid not in m:
                    print(m + ": does not match " + temp_mbid)
                continue

            print('{0:d}: {1:s}'.format(ii, m))
            # Get audio metadata
            audio_meta = audio_metadata.from_musicbrainz(m)

            vocal_instrument = []
            for a in audio_meta['artists']:
                choir_bool = a['type'] == 'vocal' and \
                             'attribute-list' in a.keys() and \
                             'choir_vocals' in a['attribute-list']
                if choir_bool:
                    vocal_instrument.append(a['attribute-list'])
                elif a['type'] in ['conductor']:
                    pass
                else:
                    vocal_instrument.append(a['type'])

            audio_meta['instrumentation_voicing'] = \
                cls.check_voice_instrumentation(vocal_instrument)

            json.dump(audio_meta, open(save_file, 'w'), indent=4)

mw = MetadataWrapper()
mw.run()

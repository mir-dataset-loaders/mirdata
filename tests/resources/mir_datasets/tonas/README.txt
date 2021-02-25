TONAS: COFLA dataset of a cappella flamenco singing 
====================================================

Compiler author of the data set
---------------------------------
COFLA (COmputational analysis of FLAmenco music) team 
http://mtg.upf.edu/research/projects/cofla
Copyright © 2012 COFLA project, Universidad de Sevilla. 
Distribution rights granted to Music Technology Group, Universitat Pompeu Fabra. 
All Rights Reserved.

Files Included
---------------------
    Audio files: 72 audio files in 16 bit mono WAV format sampled at 44.1kHz.
    Metadata: TONAS-Metadata.txt: filename (first column), title (second column) and singer (third column) for each file in Unicode UTF-8 to preserve the accents and spanish characters (ñ).

    Melodic transcriptions : 3x72 files. For each of the 72 audio files audio file we provide the following transcription files:
     
    * Instantaneous fundamental frequency (f0) and note fundamental frequency (notef0) envelopes (.f0.Corrected). These envelopes were obtained after manual edition of a f0 estimation approach based on spectral autocorrelation (Gomez and Bonada, in press), where we manually corrected some f0 errors, mainly caused by reverberation (end of phrases) and noise (background voices and percussion). We also provide notef0 values, quantized to an equal-tempered scale. Each of the 4 columns of the file correspond to:

			time_seconds energy estimatedf0_Hz estimatednotef0_Hz

    * Note transcription (.notes.Corrected): text file with note transcriptions in symbolic format. The first line contains the tuning frequency (in cents) with respect to 440 Hz. There is then one line per note, including the following information:

onset_seconds, duration_seconds, pitch_MIDI, energy

Melodic transcription
---------------------
We provide melodic transcriptions generated as follows. Three subjects participated in the transcriptions process: a musician with limited knowledge of flamenco music, an expert in flamenco music (Dr Joaquín Mora) and a flamenco singer (Cristina López). The musician first carried out detailed manual annotations. As her knowledge in flamenco music was very limited, we expected her not to use implicit knowledge on the style. Some examples were then corrected by a flamenco expert  to establish a criteria that was further applied to refine manual transcriptions. Finally, the flamenco singer independently verified the manual transcriptions.

In order to gather manual annotations, we provided the subjects with a user interface for visualizing the waveform and fundamental frequency (f0) in cents (in a piano roll representation). As transcribing everything from scratch was very time consuming, we also provided the subject with the output of a baseline transcription based on manually corrected fundamental frequency estimates and presented in (Gomez and Bonada, in press). Subjects could listen to the waveform and the synthesized transcription, while editing the melodic data until they were satisfied with the transcription. We observed that there was still a degree of subjectivity regarding the differentiation between ornaments and pitch glides.

IMPORTANT NOTE: due to the inherent subjectivity of the annotation process, different annotation methodologies would yield different transcription labellings. Our annotation represents one possible approach and we do not claim that it's an absolute ground truth. If you want to help us improve the annotation, please contact us at emilia.gomez@upf.edu. For more details on how these files were generated, we refer to the following scientific publications: 

Music material
* Mora, J., Gomez, F., Gomez, E., Escobar-Borrego, F.J., Diaz-Banez, J.M. (2010).  Melodic Characterization and Similarity in A Cappella Flamenco Cantes. 11th International Society for Music Information Retrieval Conference (ISMIR 2010).

Transcriptions
* Gomez, E., Bonada, J. (in Press). Towards computer-assisted flamenco transcription: an experimental comparison of automatic transcription algorithms from a cappella singing. Computer Music Journal. 


For more details on the methodology followed for manual annotations we refer to the following document: 

* López-Gómez, C. (2013).  Criterios para la transcripción manual de la colección de tonás, technical report, Music Technology Group, Universitat Pompeu Fabra.
http://mtg.upf.edu/node/2707

Conditions of Use
-----------------
The TONAS dataset is offered free of charge for non-commercial use only.  You can not redistribute it nor modify it. 
Dataset by COFLA team. Copyright © 2012 COFLA project, Universidad de Sevilla. 
Distribution rights granted to Music Technology Group, Universitat Pompeu Fabra.  All Rights Reserved.

Please Acknowledge TONAS in Academic Research
---------------------------------------------------
This collection was built in the context of a study on similarity and style classification of flamenco a cappella singing styles (Tonas) by the flamenco expert Dr. Joaquin Mora, Universidad de Sevilla and the COFLA group. Manual transcription were carried out by the COFLA team and Cristina López.

When TONAS is used for academic research, we would highly appreciate if scientific publications of works partly based on the TONAS dataset quote the following publication:

Music material

    Mora, J., Gomez, F., Gomez, E., Escobar-Borrego, F.J., Diaz-Banez, J.M. (2010). Melodic Characterization and Similarity in A Cappella Flamenco Cantes. 11th International Society for Music Information Retrieval Conference (ISMIR 2010).

Transcriptions

    Gomez, E., Bonada, J. (in Press). Towards computer-assisted flamenco transcription: an experimental comparison of automatic transcription algorithms from a cappella singing. Computer Music Journal.

This work was funded by the following project: 03-02-2010/03-02-2013 COFLA: Analisis Computational de la Musica Flamenca, Proyectos de Excelencia de la Junta de Andalucia, P09-TIC-4840, Junta de Andalucia (Consejeria de Innovacion, Ciencia y Empresas).

Feedback
------------------------
Problems, positive feedback, negative feedback, help to improve the annotations... it is all welcome! Please help me improve TONAS by sending your feedback to:
emilia.gomez@upf.edu AND mtg-datasets@llista.upf.edu

In case of a problem report please include as many details as possible.
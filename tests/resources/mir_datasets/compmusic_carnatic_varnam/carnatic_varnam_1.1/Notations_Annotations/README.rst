Carnatic varnam dataset is a collection of 28 solo vocal recordings, recorded for our research on intonation analysis of Carnatic raagas. The collection has the audio recordings, taala cycle annotations and notations in a machine readable format.

Download and read about it at: http://compmusic.upf.edu/carnatic-varnam-dataset
 
Please cite the following publication if you use the dataset in your work: 
    Koduri, G. K., Ishwar, V., Serrà, J., & Serra, X. (2014). Intonation analysis of rāgas in Carnatic music. Journal of New Music Research, 43(01), 73–94.


Description of the dataset
==========================


Audio music content
-------------------
They feature 7 varnams in 7 rāgas sung by 5 young professional singers who received training for more than 15 years. They are all set to Adi taala. Measuring the intonation variations require absolutely clean pitch contours. For this, all the varṇaṁs are recorded without accompanying instruments, except the drone.

Taala annotations
-----------------
The recordings are annotated with taala cycles, each annotation marking the starting of a cycle. We have later automatically divided each cycle into 8 equal parts. The annotations are made available as sonic visualizer annotation layers. Each annotation is of the format m.n where m is the cycle number and n is the division within the cycle. All m.1 annotations are manually done, whereas m.[2-8] are automatically labelled.

Notations
---------
The notations for 7 varnams are procured from an archive curated by Shivkumar (http://www.shivkumar.org/music/varnams/index.html), in word document format. They are manually converted to a machine readable format (yaml). Each file is essentially a dictionary with section names of the composition as keys. Each section is represented as a list of cycles. Each cycle in turn has a list of divisions.

Possible uses of the dataset
----------------------------
The distinct advantage of this dataset is the free availability of the audio content. Along with the annotations, it can be used for melodic analyses: characterizing intonation, motif discovery and tonic identification. The taala annotations can be used as a ground truth for rhythmic analyses such as taala recognition. The availability of a machine readable notation files allows the dataset to be used for audio-score alignment.

Availability of the dataset
---------------------------
The audio content is openly available on freesound, one can download all the files as a pack at http://www.freesound.org/people/gopalkoduri/packs/14136/. The notations and taala annotations can be downloaded from here http://compmusic.upf.edu/carnatic-varnam-dataset.

Contact
-------
If you have any questions or comments about the dataset, please feel free to write to us. 
 
Gopala Krishna Koduri
Music Technology Group,
Universitat Pompeu Fabra, 
Barcelona, Spain
gopala <dot> koduri <at> upf <dot>edu

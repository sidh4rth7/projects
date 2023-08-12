Speech Emotion Analyzer
The Speech Emotion Analyzer project strives to create an innovative machine learning model capable of discerning emotions from spoken language. By enabling emotion detection, the project aims to facilitate personalization in various aspects of daily interactions. Imagine a world where your emotions can trigger tailored recommendations for products, services, and experiences. Industries ranging from marketing to automotive could benefit immensely from such advancements.

Project Synopsis
Analyzing audio signals
Image Source: ©Fabien_Ringeval_PhD_Thesis

Datasets Explored
The project leverages two comprehensive datasets to empower emotion analysis:

RAVDESS Dataset(https://zenodo.org/record/1188976): This compilation comprises over 1500 audio files contributed by 24 diverse actors. These actors eloquently portray eight emotions, ranging from neutral and calm to happy and sad.

SAVEE Dataset(http://kahlan.eps.surrey.ac.uk/savee/Download.html): This dataset encompasses nearly 500 audio files featuring distinct male actors. Emotions are intuitively conveyed through the first two characters of each filename.

Unveiling Feature Extraction
An integral step involves extracting essential features from audio files, facilitating the model's grasp of auditory nuances. Employing the powerful LibROSA library, we execute feature extraction. Notably, all audio files are temporally adjusted to three seconds to ensure uniformity. Additionally, doubling the sampling rate augments the dataset's comprehensiveness, thereby amplifying classification accuracy.

[**LibROSA**](https://librosa.github.io/librosa/) library in python which is one of the libraries used for audio analysis. 

Crafting Robust Models
Given the classification nature of the task, the project aligns with utilizing a Convolution Neural Network (CNN). Although other models, such as Multilayer Perceptrons and Long Short Term Memory (LSTM) networks, were explored, the CNN demonstrated superior performance during evaluation.

Train vs Test
<br>
![](images/cnn.png?raw=true)
<br>

Testing with Live Voices
To validate the model's robustness against previously untested voices, we embarked on recording diverse voices articulating distinct emotions. Encouragingly, the model exhibited remarkable accuracy in identifying these novel emotional cues.

Emotion mapping:

0 - female_angry
1 - female_calm
2 - female_fearful
3 - female_happy
4 - female_sad
5 - male_angry
6 - male_calm
7 - male_fearful
8 - male_happy
9 - male_sad

Culmination and Future Prospects
The journey to construct the model was marked by iterative experimentation, diligent tuning, and rigorous training. The model boasts impeccable proficiency in distinguishing male and female voices and exhibits commendable accuracy in discerning emotions(The model was finely calibrated to achieve an impressive accuracy rate of over 70% in emotion detection. Additionally, it exhibited impeccable proficiency in gender identification, boasting a perfect accuracy rate of 100%.). The pursuit of enhanced accuracy continues as we endeavor to incorporate a broader array of audio samples into the training process.

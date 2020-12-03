import streamlit as st
import librosa
import wave
import contextlib
from tensorflow.keras import models
from VoiceRecon.features import *

st.markdown("# VoiceRecon")
st.markdown("**Predicting emotions**")

st.markdown('**VoiceRecon** is an app that uses a convoluted neural netword and the [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) data set to predict emotions.')

st.markdown('The words said in the wav file have to be one of the following two:')

st.markdown("""- Kids are talking by the door.
- Dogs are sitting by the door.""")

label = "Upload a wav file"
uploaded_file = st.file_uploader(label, type="wav", accept_multiple_files=False)


def main():
    if uploaded_file is not None:
        pipeline = models.load_model("data/70_model")
        prediction = None
        with open('user_upload.wav', mode='wb') as f:
          f.write(uploaded_file.getvalue())
        with contextlib.closing(wave.open("user_upload.wav")) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            sampling_rate = 90000/duration
            to_predict, _ = librosa.load(uploaded_file, sr = sampling_rate)
            melspectogram = create_mel_spectogram(to_predict, SR, FRAME_SIZE, HOP_SIZE, N_MELS)
            comprehensive_mfccs = create_comprehensive_MFCCS(to_predict, N_MFCCS, SR)
            chroma = create_chromas(to_predict, SR)
            full_features = create_full_features(melspectogram, comprehensive_mfccs, chroma)
            prediction = pipeline.predict(full_features)
            mx_i = ind = np.argmax(prediction)
            print(mx_i)
            df = pd.DataFrame(prediction)
            df.columns = ["Angry", "Calm", "Fearful", "Happy", "Neutral", "Sad"]
            mx = df.idxmax(axis=0)
        st.write("The emotions are: ", df)
        del pipeline

        if mx_i == 0:
            st.write("😠😠😠😠😠 RAAAWR it seems like you are angry.")
        elif mx_i == 1:
            st.write("😌😌😌😌😌 You are calm like a hindu cow.")
        elif mx_i == 2:
            st.write("😱😱😱😱😱 You are scared shitless.")
        elif mx_i == 3:
            st.write("😀😀😀😀😀 Keep being happy.")
        elif mx_i == 4:
            st.write("😐😐😐😐😐 Neutral as a swiss in WW2.")
        elif mx_i == 5:
            st.write("😭😭😭😭😭 Maybe you need to talk to someone...")


if __name__ == "__main__":
    main()


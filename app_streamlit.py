import streamlit as st
import librosa
import wave
import contextlib

st.markdown("# VoiceRecon")
st.markdown("**Predicting emotions**")

st.markdown('**VoiceRecon** is an app that uses a convoluted neural netword and the [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) data set to predict emotions.')

st.markdown('The words said in the wav file have to be one of the following two:')

st.markdown("""- Kids are talking by the door.
- Dogs are sitting by the door.""")

label = "Upload a wav file"
uploaded_file = st.file_uploader(label, type=None, accept_multiple_files=False)


def main():
    # pipeline = joblib.load('data/model.joblib')
    # print("loaded model")
    if uploaded_file is not None:

        with contextlib.closing(wave.open(uploaded_file)) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            sampling_rate = 90000/duration
            to_predict, _ = librosa.load(uploaded_file, sr = sampling_rate)


if __name__ == "__main__":
    main()

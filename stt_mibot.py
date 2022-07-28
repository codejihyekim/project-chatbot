import speech_recognition as sr


class Stt:

    def __init__(self):
        pass

    @staticmethod
    def get_audio():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            said = ""
            try:
                said = r.recognize_google(audio, language='ko-KR')
                # print(said)
            except Exception as e:
                print("다시 말 하세요" + str(e))
        return said

#
# if __name__ == "__main__":
#     Stt().get_audio()

    


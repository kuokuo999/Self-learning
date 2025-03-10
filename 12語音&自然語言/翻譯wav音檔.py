import speech_recognition as sr

r = sr.Recognizer() #建立語音辨識物件
with sr.WavFile("wav檔位置")as source:
    audio = r.record(source)
    
    print('開始翻譯')
    try:
        text = r.recognize_google(audio, language = "zh-TW")
        print(text)
    except sr.UnknownValueError:
        print("Google無法辨識此音檔")
    except sr.RequestError as e:
        print("無法由Google Speech Recognition取得結果; {0}".format(e))
    print('翻譯結束')
        

    
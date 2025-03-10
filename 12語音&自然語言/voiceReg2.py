from pydub import AudioSegment
from pydub.utils import make_chunks
import speech_recognition as sr
import shutil, os
 
os.mkdir('temdir')
audiofile = AudioSegment.from_file("python1.wav", "wav")  #讀語音檔
chunklist = make_chunks(audiofile, 30000)  #切割語音檔
#儲存分割後的語音檔
for i, chunk in enumerate(chunklist):  
    chunk_name = "temdir/chunk{0}.wav".format(i)
    print ("存檔：", chunk_name)
    chunk.export(chunk_name, format="wav")

r = sr.Recognizer()  #建立語音辨識物件
print('開始翻譯...')
file = open('python1_sr.txt', 'w')  #儲存辨識結果
for i in range(len(chunklist)):
    try:
        with sr.WavFile("temdir/chunk{}.wav".format(i)) as source: 
            audio = r.record(source)
        result = r.recognize_google(audio, language="zh-TW")  #辨識結果
        print('{}. {}'.format(i+1, result))
        file.write(result)
    except sr.UnknownValueError:
        print("Google Speech Recognition 無法辨識此語音！")
    except sr.RequestError as e:
        print("無法由 Google Speech Recognition 取得結果; {0}".format(e))
file.close()
print('翻譯結束！')
shutil.rmtree('temdir')  #移除分割檔
   
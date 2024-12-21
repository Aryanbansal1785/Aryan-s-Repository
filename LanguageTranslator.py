from tkinter import * #tkinter is GUI library . Python when combined with tkinter creates easy and fast GUI appls
from tkinter import ttk #ttk is used to style tkinter widgets like bg color or font style
from googletrans import Translator , LANGUAGES
import speech_recognition as sr
import pyttsx3  #text to speech library


root = Tk()   #root is an instance of the Tk class, which represents the main window, or the root window, of your application.
root.geometry ('1100x320')
root.resizable(0,0)
root['bg']= 'skyblue'

root.title('Language Translator by Aryan')
Label(root, text = "Language Translator", font = 'Arial 20 bold').pack()


Label(root,text = "Enter Text" , font= "Transitional 13 bold", bg = "white smoke").place(x=165,y=90)

Input_text = Entry(root,width=60)
Input_text.place(x=30,y=130)
Input_text.get()



Label(root,text = "Output" , font= "Transitional 13 bold", bg = "white smoke").place(x=780,y=90)
Output_text = Text(root, font="Arial 18 bold ", height=5, wrap = WORD, padx=5,pady=5,width=50)
Output_text.place(x=600,y=130)

language = list(LANGUAGES.values())
dest_lang = ttk.Combobox(root, values= language, width=22 ) #Combobox (drop-down menu) for selecting the destination language.
dest_lang.place(x=130,y=160)
dest_lang.set('choose language')


def Translate():    # This function translates the input text to the selected language using Google Translate API and updates the output text widget.
    translator = Translator()
    translated = translator.translate(text=Input_text.get(), dest=dest_lang.get())
    Output_text.delete(1.0,END)
    Output_text.insert(END, translated.text)
    
trans_btn = Button(root,text='Translate', font='arial 12 bold',pady=5,command=Translate, bg='orange',activebackground='green')
trans_btn.place(x=445,y=180)



def SpeechInput():
    r= sr.Recognizer()
    with sr.Microphone() as source: #microphone class used from sr library to capture audio input from user
        print("Speak something....") 
        audio=r.listen(source)
        
        
    try:   #We use a try block to handle potential errors during the speech recognition process.
        print("Recognizing...")
        spoken_text=r.recognize_google(audio)
        Input_text.delete(0,END)  #This line deletes the content of the input text field. The index 0 represents the start of the text field, and END represents the end of the text field. By deleting from index 0 to END, the entire content of the text field is cleared.
        Input_text.insert(0,spoken_text) #This line inserts the recognized spoken text into the input text field at index 0, effectively replacing any existing text in the field. spoken_text contains the text recognized from the user's speech input.
    
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        
speech_btn = Button(root, text='Speech Input', font='arial 12 bold', pady=5, command=SpeechInput, bg='orange',activebackground='green')
speech_btn.place(x=330, y=180)

trans_btn = Button(root, text='Translate', font='arial 12 bold', pady=5, command=Translate, bg='orange',activebackground='green')
trans_btn.place(x=445, y=180)
    
root.mainloop()

import speech_recognition as sr
import pyttsx3 
import pywhatkit # it is one of the most popular library for WhatsApp and YouTube automation
import datetime 
import wikipedia 
import webbrowser


 

listener = sr.Recognizer()
machine = pyttsx3.init()  #python text to speech library initialised with a variable machine


def talk(text):
    machine.say(text) #This line instructs the text-to-speech engine to "say" the specified text (text variable).
    machine.runAndWait() #This line blocks the execution of the program until all currently queued commands in the text-to-speech engine are processed and spoken. It ensures that the program does not proceed to the next line of code until the text has been completely spoken.
 
    
def get_instruction_and_check_jarvis():
    instruction = ""
    try:   #question if our microphone doesn't work or we face any other problem
        with sr.Microphone() as source:
            print("listening....")
            listener.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            audio = listener.listen(source)
            instruction = listener.recognize_google(audio) #creating a variable instruction and recognizing origin voice and by google api converting it to text
    except sr.UnknownValueError: #it means python will not do anything if any exception occurs
            print("Sorry, I didn't catch that. Please repeat.")
    except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
          
    return instruction.lower().strip() # strip removes leading and trailing whitespace (spaces, tabs, and newline characters) from a string

def play_Jarvis():     #function created for more functionalities like playing a video from youtube
    stop_commands = ['stop','thankyou']
    while True:
        instruction = get_instruction_and_check_jarvis()
        print(instruction)
        
        if "play" in instruction:
            song = instruction.replace('play','').strip()
            talk("Searching and playing" + song)
            pywhatkit.playonyt(song) 
            
            # Use webbrowser to open the YouTube search results for the song
            search_query = f"https://www.youtube.com/results?search_query={song}"
            webbrowser.open(search_query)
            
        elif any(stop_command in instruction for stop_command in stop_commands):
            talk('Goodbye!')
            break  # exit the loop and stop the program
            
        elif 'time' in instruction:
            time= datetime.datetime.now().strftime('%I:%M%p')
            talk ('Current Time' + time)
            
        elif 'date' in instruction:
            date=datetime.datetime.now().strftime('%m/%d/%y')
            talk('today date'+ date)
            
        elif 'how are you?' in instruction:
            talk('I am fine, how about you')
            
        elif 'what is your name?' in instruction:
            talk ('I am Jarvis,What can i do for you?')
        
        elif 'who is' in instruction:
            human = instruction.replace('who is', '')
            info = wikipedia.summary(human)
            print(info)
            talk (info)
            
        
    
        else:
            talk('Please Repeat')    
            
if __name__ == "__main__":  #This condition checks whether the script is being run as the main program or if it is being imported as a module into another script.
    play_Jarvis()
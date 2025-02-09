def chatbot():
    while True:
        user_input=input("Enter any thing ")
        if(user_input=="hello"):
            print("hello! how can i help you ")
        elif(user_input=="stop"):
            print(" chat is stop  ")
        elif(user_input=="how many types of books"):
            print("types of books \n Horror\n Mystery")
        elif(user_input=="which one i read "):
            print("Horror")
        elif(user_input=="what is the chatbot"):
            print("chatbot is your assistant ")
        elif(user_input=="chatbot is freee "):
            print("yes ")
        elif(user_input=="whats is the current weather "):
            print(" I dont know")

        else:
             print("unexpected sentence")
chatbot()


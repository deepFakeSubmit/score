import os
import urllib.request


if __name__== "__main__":

    # train gmm models
    print("# Train gender models")
    os.system('python3 ModelsTrainer.py')

    # test system and recognise/identify Spoof Speech
    print(" # Identify genders")
    os.system('python3 Identifier.py')

import cv2
import os
import collections

from difflib import SequenceMatcher

from Modules import DetectPlates
from Modules import DetectChars

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)


def similar(a, b):
    return round(SequenceMatcher(None, a, b).ratio(), 2)


def main(fileName):

    image = cv2.imread(fileName)

    if image is None:
        print("Error: Image Not found \n")
        return

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(image)
    listOfPossiblePlatesChars = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    if len(listOfPossiblePlatesChars) == 0:
        print("Warning: No license plates were detected")
        return ""

    else:
        listOfPossiblePlatesChars.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        licPlate = listOfPossiblePlatesChars[0]

        if len(licPlate.strChars) == 0:
            print("Warning: No characters were detected")
            return

    return licPlate.strChars


if __name__ == "__main__":

    if not DetectChars.loadKNNDataAndTrainKNN():
        print("Error: KNN traning was not successful\n")

    size = len(os.listdir("Dataset/Dataset 1"))

    accuracy = []
    count = 0
    charData = {}
    char1Data = {}


    for fileName in os.listdir("Dataset/Dataset 1"):
        print(fileName)
        recognizeText = main("Dataset/Dataset 1/" + fileName)
        acutalText = os.path.splitext(fileName)[0]
        accuracy.append(similar(acutalText, recognizeText))
        count = count + 1
        print(str(count) + "/" + str(size) + " => " + "(Recognized Plate Value = " + recognizeText + ")" + " (Actual Plate value = " + acutalText + ")") #Accuracy = " + str(similar(acutalText, recognizeText) * 100) + "%

        chars = list(recognizeText)


        for i in range(len(chars)):
            if charData.__contains__(chars[i]):
                charData[chars[i]] = charData[chars[i]] + i
            else:
                charData[chars[i]] = 1

        chars1 = list(acutalText)
        for i in range(len(chars1)):
            if char1Data.__contains__(chars1[i]):
                char1Data[chars1[i]] = char1Data[chars1[i]] + i
            else:
                char1Data[chars1[i]] = 1


    print("\n Accuracy of the algorithm is " + str(sum(accuracy) / len(accuracy) * 100)+"\n")
    print ("character in recognized text")
    print(collections.OrderedDict(charData))
    print ("characters in actual text")
    print(collections.OrderedDict(char1Data))

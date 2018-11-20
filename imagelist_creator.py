# -*- coding utf-8 -*-

from xml.dom import minidom

import os
import cv2
import numpy as np

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class imList(object):

    def __init__(self, pathToDir, xmlfile):
        self.pathToDir = pathToDir
        #self.listOfFiles = self.getListOfFiles()
        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(self.pathToDir):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        self.listOfFiles = listOfFiles
        print(color.RED + color.BOLD + xmlfile + ': ' + str(len(self.listOfFiles)) + ' patterns' + color.END)
        save_path_file = self.saveToXml(xmlfile)
        self.fNum, self.pathToImage, self.sizeFNum = self.readFromXml( save_path_file)

        print (color.BLUE + color.BOLD + str(self.sizeFNum) + color.END, self.fNum)


    def getListOfFiles(self):
            # create a list of file and sub directories
            # names in the given directory
        listOfFile = os.listdir(self.pathToDir)
        allFiles = list()
            # Iterate over all the entries
        for entry in listOfFile:
                # Create full path
            fullPath = os.path.join(self.pathToDir, entry)
                # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
            return allFiles

    def saveToXml(self, xmlfile):

        root = minidom.Document()
        xml = root.createElement('opencv_storage')
        root.appendChild(xml)
        images = root.createElement('images')
        images.setAttribute('id', 'c1 - left')
        xml.appendChild(images)
        for num in range(0, len(self.listOfFiles)):
            images.appendChild(root.createTextNode(self.listOfFiles[num]))

        xml_str = root.toprettyxml(indent="\t")
        save_path_file = xmlfile + ".xml"
        with open(save_path_file, "w") as f:
            f.write(xml_str)
        print (color.GREEN + color.BOLD +  "-------------"  + save_path_file + " is done" + color.END)
        return save_path_file

    def readFromXml(self, save_path_file):
        def file_len(fname):
            with open(fname) as f:
                for i, l in enumerate(f):
                    pass
            return i+1
        i = file_len(save_path_file)
        fNum = []
        pathToImage = []
        with open(save_path_file, "r") as f:
            for cnt, line in enumerate(f):
                if cnt > 2 and cnt < i-2 :
                    l = line.replace("\t", "").replace('\n', '')
                    pathToImage.append(l)
                    a = os.path.basename(l)
                    a = os.path.splitext(a)[0]
                    fNum.append(int(a))
        return fNum, pathToImage, np.size(fNum)

def createTxtFile(txtfile, data):
    f = open(txtfile,"w+")
    for i in range(len(data)):
        #print(i, len(data))
        f.write("%d\n" % data[i])
    f.close()


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        com = a_set & b_set
        print("Number of common pattern in C1 and C2: ", len(com))

    else:
        print("No common elements")

    return list(set(com))
def main():

    # Get the list of all files in directory tree at given path
    #__mtL = imList(pathToDir='C:/Working/Skagen/Calibration/framesL/',
    #              xmlfile = 'left')
    __mtL = imList(pathToDir='C:/Working/Skagen/Calibration/Frames/C1/', xmlfile = 'left_full')

     #              __mtR = imList(pathToDir='C:/Working/Skagen/Calibration/framesR/',
     #             xmlfile = 'right')
    __mtR = imList(pathToDir='C:/Working/Skagen/Calibration/Frames/C2/', xmlfile = 'right_full')
    #listOfFiles = getListOfFiles(dirName)
    #print(__mtL.fNum)
    com = common_member(a = __mtL.fNum, b= __mtR.fNum)
    createTxtFile(txtfile="common_pattern.txt", data=com)

# print("xml list is done")






if __name__ == '__main__':
    main()
    print ("done")
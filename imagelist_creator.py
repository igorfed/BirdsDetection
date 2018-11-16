# -*- coding utf-8 -*-

from xml.dom import minidom

import os

class imList(object):

    def __init__(self, pathToDir, xmlfile):
        self.pathToDir = pathToDir
        #self.listOfFiles = self.getListOfFiles()
        listOfFiles = list()
        for (dirpath, dirnames, filenames) in os.walk(self.pathToDir):
            listOfFiles += [os.path.join(dirpath, file) for file in filenames]
        self.listOfFiles = listOfFiles
        print (xmlfile + ': ' + str(len(self.listOfFiles)) + ' patterns')
    # Print the files
        #for elem in listOfFiles:
        #    print(elem)

        self.saveToXml(xmlfile)
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
        for num in range(0, len(self.listOfFiles)-1):
            images.appendChild(root.createTextNode(self.listOfFiles[num]))

        xml_str = root.toprettyxml(indent="\t")
        save_path_file = xmlfile + ".xml"
        with open(save_path_file, "w") as f:
            f.write(xml_str)
        print("-------------" + save_path_file + " is done")

def main():

    # Get the list of all files in directory tree at given path
    __mtL = imList(pathToDir='C:/Working/Skagen/Calibration/framesL/',
                  xmlfile = 'left')
    __mtR = imList(pathToDir='C:/Working/Skagen/Calibration/framesR/',
                  xmlfile = 'right')

    #listOfFiles = getListOfFiles(dirName)


   # print("xml list is done")






if __name__ == '__main__':
    main()
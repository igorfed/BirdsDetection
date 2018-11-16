# -*- coding utf-8 -*-

from xml.dom import minidom
# minidome use to xml library

import os

'''
    For the given path, get the List of all files in the directory tree 

'''


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def main():
    dirName = 'C:\Working\Skagen\Calibration\L'

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    root = minidom.Document()
    xml = root.createElement('opencv_storage')
    #xml.setAttribute('id', 'version-id')
    root.appendChild(xml)
    images = root.createElement('images')
    images.setAttribute('id', 'c1')
    xml.appendChild(images)

    for num in range(0,10):
       # productchild=root.createElement(str(listOfFiles[num]))
        images.appendChild(root.createTextNode(listOfFiles[num]))
        #productchild.setAttribute('id', 'test-id')
        #xml.appendChild(productchild)
    xml_str = root.toprettyxml(indent="\t")
    save_path_file = "test.xml"
    with open(save_path_file, "w") as f:
        f.write(xml_str)

    # Print the files
    for elem in listOfFiles:
        print(elem)

    print("****************")

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # Print the files
    for elem in listOfFiles:
        print(elem)


if __name__ == '__main__':
    main()
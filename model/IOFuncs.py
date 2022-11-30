import os
import argparse
import sys
import misc
from PIL import Image

rootDir = r"C:\Users\Garrett London\PycharmProjects\VisionFinal"
dict = {'C':'Cars','F':'Faces','L':'Leaves','M':'Motorcycles','A':'Planes'}
rev_dict = {'Cars':'C','Faces':'F','Leaves':'L','Motorcycles':'M','Planes':'A'}
img_dir = 'Test'
out_path = 'model/output.txt'
def run_model(mx,path):
    img = Image.open(path)
    results = [(label, score) for label, score in mx.predict(img).items()]
    results = sorted(results, key=lambda k: -k[1])
    result, conf = results[0]
    return result

def get_input(path, mx):
    fin = open(path,'r')
    fout = open(out_path, 'a')
    imagelist = fin.readlines()
    for i in imagelist:
        classifier = dict[i[:1]]
        imgfilename = i[1:-1]
        tempdir = os.path.join(img_dir,classifier)
        result = run_model(mx, os.path.join(tempdir, imgfilename))
        fout.write(f"{i[:-1]} - {result}\n")
    fin.close()
    fout.close()




if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", type=ascii, required=False, help="Specify path of input directory")
    args = vars(ap.parse_args())
    dir_path = args["path"]
    if dir_path is None:
        sys.exit("Must specify path")
    dir_path = dir_path[1:-1]

    contentPath = os.path.join(rootDir,'model')
    modelPath = os.path.join(contentPath, 'saved_model.pth')
    inputFilePath = os.path.join(rootDir, dir_path)
    classes = ['Car', 'Face', 'Leaf', 'Motorcycle', 'Airplane']

    '''
    file = open(inputFilePath,'w')
    testImagesPath = os.path.join(rootDir, 'Test')
    for i in os.listdir(testImagesPath):
        currdir = os.path.join(testImagesPath,i)
        for j in os.listdir(currdir):
            str = f"{rev_dict[i]}{j}"
            file.write(f"{str}\n")
    file.close()
    '''
    mx = misc.Model.from_file(labels=classes, path=modelPath)
    get_input(inputFilePath,mx)
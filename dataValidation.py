import os
import shutil
files = ['\\Annotations', '\\Raw Videos']

#get files in Annotations
cd = os.getcwd()

paths = [cd + i for i in files]

#list of files in annotations
annotations = os.listdir(paths[0])
videos = os.listdir(paths[1])

annotationNames = [annotation[0:len(annotation)-4] for annotation in annotations]
videoNames = [video[0:len(video)-4] for video in videos]

test = [i in videoNames for i in annotationNames]
present = list(filter(lambda x: x in videoNames,annotationNames))

matchedAnnotations = ['Annotations\\' + i + '.xml' for i in present]
matchedVideos = ['Raw Videos\\' + i + '.mov' for i in present]
pathsFrom = matchedAnnotations + matchedVideos


pathAnnotations = ['Model Data\\' + i +'.xml' for i in present]
pathVideos = ['Model Data\\' + i + '.mov' for i in present]
pathsTo = pathAnnotations + pathVideos
for i in range(0,len(pathsTo)):
    print(i)
    shutil.copy(pathsFrom[i],pathsTo[i])

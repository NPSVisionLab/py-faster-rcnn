
import sys
import os
import numpy as np
import shutil
from collections import namedtuple
import re
import tempfile


trainFile = "trainFiles.txt"
valFile = "validateFiles.txt"

Score = namedtuple('Score', 'tp tn fp fn')
Result = namedtuple('Result', 'recall tnr score')

class ScoreKeeper:
    def __init__(self):
        self.scores = []
        self.results = []

    def addScore(self, log):
        iterLabelEx = re.compile(".*?, label = (\d+)")
        iterOutputEx = re.compile(".*?, output = (\d+)")
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        logf = open(log, "r")
        if not logf:
            raise RuntimeWarning("Could not open log " + log)
        for line in logf:
            iterLabel = iterLabelEx.findall(line)
            if len(iterLabel) > 0:
                lastLabel = int(iterLabel[0])
            iterOutput = iterOutputEx.findall(line)
            if len(iterOutput) > 0:
                output =  (int)(iterOutput[0])
                if (lastLabel == output):
                    if lastLabel == 1:
                        tp += 1

                    else:
                        tn += 1
                else:
                    if lastLabel == 1:
                        fn += 1
                    else:
                        fp += 1
        score = Score(tp, tn, fp, fn)
        logf.close()
        self.scores.append(score)

        r_recall = float(score.tp) / (score.tp + score.fn)
        r_tnr = float(score.tn) / (score.tn + score.fp)
        r_score = float((r_recall + r_tnr)) / 2
        result = Result(r_recall, r_tnr, r_score)
        self.results.append(result)

    def __str__(self):
        cnt = len(self.results)
        total = 0
        for result in self.results:
            total += result.score
        res = "Total score = {0}\n".format(total / cnt)

        for result in self.results:
            res += 'recall={0} tnr={1} score={2}\n'.format(result.recall, result.tnr, result.score)

        return res



def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def buildFileLists(traindir, dir, count, size, labelstr):
    filelist = get_filepaths(dir)
    valList = filelist[count:count+ size]
    if count > 0:
        trainList = filelist[0:count]
        trainList.extend(filelist[count+ size:])
    else:
        trainList = filelist[count+size:]
    tf = open(os.path.join(traindir, trainFile), "a+")
    if tf == None:
        RuntimeError("Could not create train list file")
    vf = open(os.path.join(traindir, valFile), "a+")
    if vf == None:
        RuntimeError("Could not create validate list file")
    for f in trainList:
        tf.write(f + " " + labelstr + "\n")
    tf.close()
    for f in valList:
        vf.write(f + " " + labelstr + "\n")
    vf.close()

def buildLMDB(traindir, posdir, negdir, poscount, possize, negcount, negsize):
    try:
        os.remove(os.path.join(traindir, trainFile))
    except OSError:
        pass
    try:
        os.remove(os.path.join(traindir, valFile))
    except OSError:
        pass
    try:
        shutil.rmtree(os.path.join(traindir, "train_lmdb"))
    except Exception:
        pass
    try:
        shutil.rmtree(os.path.join(traindir, "validate_lmdb"))
    except Exception:
        pass

    buildFileLists(traindir, posdir, poscount, possize, "1")
    buildFileLists(traindir, negdir, negcount, negsize, "0")
    print("building training database")
    os.system(
	 "/home/trbatcha/caffe/build/tools/convert_imageset --shuffle --resize_width 255 --resize_height 255  / "  +
     trainFile + " " + traindir + '/train_lmdb')
    print("building validate database")
    os.system(
	 "/home/trbatcha/caffe/build/tools/convert_imageset --shuffle --resize_width 255 --resize_height 255  / "  +
     valFile + " " + traindir + '/validate_lmdb')


def change_solver(solver, newsolver, property, value, quotes = False):
    print('comparing {0} to {1}'.format(solver, newsolver))

    if solver == newsolver:
        tf = tempfile.NamedTemporaryFile(dir = os.path.dirname(solver))
        wfile = tf.name
    else:
        wfile = newsolver
    rfp = open(solver, "r")
    if rfp == None:
        print("Could not open solver {0}".format(solver))
    wfp = open(wfile, "w+")
    if wfp == None:
        print("Could not open newsolver {0}".format(wfile))
    line = rfp.readline()
    while line:
        prop = line.split(':')
	if prop[0] == property:
	    if quotes:
		wfp.write(property + ': "{0}"\n'.format(value))
	    else:
		wfp.write(property + ': {0}\n'.format(value))
	else:
	    wfp.write(line)
        line = rfp.readline()
    rfp.close()
    wfp.close()
    if solver == newsolver:
        print('Renaming {0} to {1}'.format(wfile, newsolver))
        os.rename(wfile, newsolver)

def trainInc(traindir, posdir, negdir, poscount, possize, negcount, negsize, solver, weights, log):

    buildLMDB(traindir, posdir, negdir, poscount, possize, negcount, negsize)
    print("Training using valiation samples starting at {0} of size {1}".format(poscount, possize))
    os.system(
	 "/home/trbatcha/caffe/build/tools/caffe train -gpu 0 -solver "  +
              solver + " -weights " + weights + " >> " + log + " 2>&1")

def detect(traindir, detector, weights, poscount, possize, count, log):
    print("Run detection on validation set starting at {0} of size {1}".format(poscount, count))
    os.system(
	 "/home/trbatcha/caffe/build/tools/caffe test -gpu 0 -model "  +
              detector + " -weights " + weights + " -iterations " + str(count) + " > " + log + " 2>&1")



def runTraining(traindir, posdir, negdir, solver, weights, log, dlog, scount):

    posList = get_filepaths(posdir)
    negList = get_filepaths(negdir)
    possize = len(posList) / 10
    negsize = len(negList) / 10
    max_count = int(scount)
    for count in xrange(max_count):
        base_lr = 10 ** np.random.uniform(-6,-2)
        momentum = np.random.uniform(0.1, 0.7)
        nextSolver = "solver_{0}_{1}.prototxt".format(base_lr, momentum)
        change_solver(solver, nextSolver, "base_lr", base_lr)
        change_solver(nextSolver, nextSolver, "momentum", momentum)
        nextlog = "{0}_lr_m_{1}_{2}.log".format(log, base_lr, momentum)
        trainInc(trainDir, posdir, negdir, 0, possize, 0, negsize, nextSolver, weights, nextlog)
        nextdlog = "{0}_lr_m_{1}_{2}.log".format(dlog, base_lr, momentum)
        detect(trainDir, "deploy.prototxt", "snapshots/snap__iter_2000.caffemodel", 0, possize, possize + negsize, nextdlog)
        score = ScoreKeeper()
        score.addScore(nextdlog)
        print("Score for base_lr= {0}, momentum= {1}".format(base_lr, momentum))
        print(score)
        with open(nextlog, 'a') as f:
            f.write("score for base_lr = {0}, momentum = {1}\n".format(base_lr, momentum))
            f.write(score.__str__())
        os.remove(nextdlog)


def detectInc(trainDir, posdir, negdir, poscount, possize, negcount, negsize, log, score):
    try:
        os.remove(os.path.join(trainDir, trainFile))
    except OSError:
        pass
    try:
        os.remove(os.path.join(trainDir, valFile))
    except OSError:
        pass

    buildFileLists(trainDir, posdir, poscount, possize, "1")
    buildFileLists(trainDir, negdir, negcount, negsize, "0")
    weightFile = "snapshots/model{0}.caffemodel".format(poscount)
    if (os.path.exists(weightFile) == False):
        print("Weight file {0} does not exist".format(weightFile))
    else:
        detect(trainDir, "deploy.prototxt", weightFile, poscount, possize, possize + negsize, log)
    score.addScore(log)

def runDetection(trainDir, posdir, negdir, log, score):
    posList = get_filepaths(posdir)
    negList = get_filepaths(negdir)
    possize = len(posList) / 10
    negsize = len(negList) / 10
    for i in range(0,9):
        nextdlog = "{0}_{1}.log".format(log, i*possize)
        try:
            os.remove(os.path.join(trainDir, nextdlog))
        except OSError:
            pass
        detectInc(trainDir, posdir, negdir, i*possize, possize, i*negsize, negsize, nextdlog, score)
    lastpsize = len(posList) - 9*possize
    lastnsize = len(negList) - 9*negsize
    detectInc(trainDir, posdir, negdir, 9*possize, lastpsize, 9*negsize, lastnsize, nextdlog, score)

'''
This program assumes its being run from the CVAC root directory
'''
if __name__ == "__main__":
    '''
    usage:
    program trainDir posdir negdir num_of_searches
    '''
    trainDir = sys.argv[1]
    posDir = sys.argv[2]
    negDir = sys.argv[3]
    count = sys.argv[4]
    #solver= sys.argv[4]
    #weights= sys.argv[5]
    #log= sys.argv[6]
    os.chdir(trainDir)
    solver =  "quick_solver.prototxt"
    weights = os.path.join("..", "bvlc_googlenet.caffemodel")
    log = "searchLog"
    dlog = "detectLog"

    # Uncomment to run both the trainging and detection
    runTraining(trainDir, posDir, negDir, solver, weights, log, dlog, count)
    # Uncomment to only run the detection on what was trained previously
    #runDetection(trainDir, posDir, negDir, dlog, score)
    #print(score)
       

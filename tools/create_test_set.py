
import sys
import getopt
import os
import random
import shutil

if __name__ == '__main__':
    src_dir = ''
    dest_dir = ''
    percent = 0.0
    argv = sys.argv[1:]

    try:
        opts,args = getopt.getopt(argv, "s:d:p:")
    except getopt.GetoptError:
        print (sys.argv[0] + "-s <src_dir> -d <dest_dir> -p <percent>")
        sys.exit(2)

    for opt,arg in opts:
        if opt == '-s':
            src_dir = arg
        elif opt == '-d':
            dest_dir = arg
        elif opt == '-p':
            percent = float(arg)
    if src_dir == '':
        print("Need to pass source image directory")
        sys.exit(1)

    if dest_dir == '':
        print("Need to pass destination image directory")
        sys.exit(1)

    if not os.path.exists(src_dir):
        print("Source directory does not exist")
        sys.exit(1)
    if not os.path.exists(dest_dir):
        print("Creating destination directory " + dest_dir)
        os.makedirs(dest_dir)
    src_file_list =   [x for x in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, x))]
    move_cnt = int(len(src_file_list) * percent)
    print("Moving " + str(move_cnt) + " files to " + dest_dir)
    toMove = []
    for i in range(move_cnt):
        next = random.choice(src_file_list)
        toMove.append(next)
        src_file_list.remove(next)
    for next in toMove:
        shutil.move(os.path.join(src_dir,next), os.path.join(dest_dir, next))



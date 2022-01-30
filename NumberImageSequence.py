
import argparse
import cv2
import glob
import numpy as np
import os
import sys
import time

def get_filename_parts(fn):
    s = os.path.split(fn)
    if ( s[0] == '' ):
        s[0] = '.'

    e = os.path.splitext(s[1])

    return [ s[0], e[0], e[1] ]

def load_image(fn):
    if ( not os.path.isfile( fn ) ):
        raise Exception('%s does not exist. ' % ( fn ))

    return cv2.imread( fn, cv2.IMREAD_UNCHANGED )

def add_string_line_2_img(img, s, hRatio, bgColor=(70, 30, 10), fgColor=(0,255,255)):
    """
    Add a string on top of img.
    s: The string. Only supports 1 line string.
    hRatio: The non-dimensional height ratio for the font.
    """

    assert( hRatio < 1 and hRatio > 0 )

    H, W = img.shape[:2]

    # The text size of the string in base font.
    strSizeOri, baselineOri = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)

    # The desired height.
    hDesired = np.ceil( hRatio * H )
    strScale = hDesired / strSizeOri[1]

    # The actual font size.
    strSize, baseline = \
        cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, strScale, 1 )

    fontSize, baseLineFont = \
        cv2.getTextSize("a", cv2.FONT_HERSHEY_SIMPLEX, strScale, 1 )

    # Draw the box.
    hBox = strSize[1] + fontSize[1]
    wBox = strSize[0] + 2 * fontSize[0]

    img = img.copy()
    pts = np.array([[0,0],[wBox,0],[wBox,hBox],[0,hBox]],np.int32)
    cv2.fillConvexPoly( img, pts, color=bgColor )
    cv2.putText(img, s, (fontSize[0], int(hBox-fontSize[1]/2.0)),
        cv2.FONT_HERSHEY_SIMPLEX, strScale, color=fgColor, thickness=1)

    return img

def read_and_mark_single_image(fn, newSize):
    '''
    fn (string): The filename.
    newSize (2-element): The new size [W, H] of the marked image.
    '''

    img = load_image(fn)
    img = cv2.resize( img, (newSize[0], newSize[1]), interpolation=cv2.INTER_LINEAR )

    # Get the filename.
    name = os.path.basename(fn)

    # Annotate.
    img = add_string_line_2_img( img, name, 0.05 )

    return img

def string_2_int(s, expected, delimiter=','):
    ss = s.split(delimiter)

    if ( len(ss) != expected ):
        raise Exception('Expecting %d integers from %s. ' % ( expected, s ))

    ints = []
    for item in ss:
        ints.append( int( item.strip() ) )

    return ints

def find_files(path, pattern):
    if ( 2 == sys.version_info.major ):
        s = os.path.join( path, pattern )
        files = sorted( glob.glob( s ) )
    else:
        s = '%s/**/%s' % ( path, pattern )
        files = sorted( glob.glob( s, recursive=True ) )

    if ( 0 == len(files) ):
        raise Exception('No files found with %s. ' % ( s ))

    return files

def get_max_number(nFiles, maxNum):
    if ( maxNum > 0 ):
        if ( maxNum <= nFiles ):
            return maxNum
    else:
        return nFiles

def handle_args():
    parser = argparse.ArgumentParser(description='Add marks to a sequence of image and make a video. ')

    parser.add_argument('indir', type=str, 
        help='The input directory. ')

    parser.add_argument('--pattern', type=str, default='*.png', 
        help='The search pattern of the input images. ')

    parser.add_argument('--new-size', type=str, default='320, 240', 
        help='The new image size [W, H]. ')

    parser.add_argument('--fps', type=float, default=2.0, 
        help='The fps of the output video. ')

    parser.add_argument('--max-num', type=int, default=0, 
        help='The maximum number of images to process. Debug use, set 0 to disable. ')

    args = parser.parse_args()

    return args

def main():
    print('Hello, %s! ' % ( os.path.basename(__file__) ))

    args = handle_args()

    newSize = string_2_int(args.new_size, 2)
    print('newSize = {}'.format(newSize))

    # Get the file list and figure out how many files to process.
    files0 = find_files( os.path.join(args.indir, 'left'),  args.pattern )
    files1 = find_files( os.path.join(args.indir, 'right'), args.pattern )

    assert( len(files0) == len(files1) )

    nFiles = get_max_number( len(files0), args.max_num )
    print('%d files to process. ' % ( nFiles ))

    # The video writer.
    videoFn = os.path.join( args.indir, 'Marked.mp4' )
    vw = cv2.VideoWriter(videoFn, cv2.VideoWriter_fourcc(*'MP4V'), args.fps, (newSize[0]*2, newSize[1]))

    startTime = time.time()
    for i in range(nFiles):
        fn0 = files0[i]
        fn1 = files1[i]
        print('%d/%d: %s' % (i+1, nFiles, fn0))
        img0 = read_and_mark_single_image(fn0, newSize)
        img1 = read_and_mark_single_image(fn1, newSize)
        img  = np.concatenate((img0, img1), axis=1)
        vw.write( img )

    vw.release()
    endTime = time.time()

    print('Done with processing %d files. ' % (nFiles))
    print('Total time is %fs. ' % ( endTime - startTime ))

    return 0

if __name__ == '__main__':
    sys.exit( main() )

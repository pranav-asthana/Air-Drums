#importing modules
import cv2
import numpy as np
import math
import time
import _thread
import wave
import struct

def playSound(name):
    import simpleaudio as sa

    wave_obj = sa.WaveObject.from_wave_file(name)
    play_obj = wave_obj.play()


    ####CRASHES ON FAST INPUT####
    # import pyglet
    # player = pyglet.media.Player()
    # src = pyglet.media.load(name)
    # player.volume = 0.1
    # player.queue(src)
    # player.play()

    #####VERY SLOW####
    # import pygame.mixer
    # pm = pygame.mixer
    # pm.init()
    # sound = pm.Sound(name)
    # sound.set_volume(0.5)
    # sound.play()



def drawEllipse(contours, text):
    if(contours == None or len(contours) == 0):
        return ((-100,-100), None)
    c = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    if(cv2.contourArea(c) < 500):
        return ((-100,-100), None)
    ellipse = cv2.fitEllipse(c)
    cv2.ellipse(img, ellipse, (0,0,0), 2)

    blank = np.zeros(img.shape[0:2])
    ellipseImage = cv2.ellipse(blank, ellipse, (255, 255, 255), -2)
    # cv2.imshow("ell",ellipseImage)

    M = cv2.moments(c)
    if M["m00"] == 0:
        return
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    if radius > 10:
        # draw the ellipse and centroid on the frame,
        # then update the list of tracked points
        # cv2.circle(img, (int(x), int(y)), int(radius),(0, 0, 0), 2)
        cv2.circle(img, center, 3, (0, 0, 255), -1)
        cv2.putText(img,text, (center[0]+10,center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0),2)
        cv2.putText(img,"("+str(center[0])+","+str(center[1])+")", (center[0]+10,center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0),1)

    return (center, ellipseImage)

def detectCollision(imgA, imgB, velocity, touching, name):
    mA = cv2.moments(imgA, False)
    mB = cv2.moments(imgB, False)
    blank = np.zeros(img.shape[0:2])
    if type(imgA) == type(None) or type(imgB) == type(None):
        return
    intersection = cv2.bitwise_and(imgA, imgB)
    area = cv2.countNonZero(intersection)
    if area < 20:
        touching = False
    if area > 100 and not touching:
        # print(int(mA["m01"] / mA["m00"])< int(mB["m01"] / mB["m00"]))
        # print(area)
        if int(mA["m01"] / mA["m00"])< int(mB["m01"] / mB["m00"]):
            if velocity > 10:
                _thread.start_new_thread(playSound, (name,))
                # playSound(name)
        touching = True
    return touching

#capturing video through webcam
cap=cv2.VideoCapture(0)
frameCount = 0
timeStart = time.time()

b1 = (0,0)
b2 = (0,0)
currentBlueVelocity = 0
r1 = (0,0)
r2 = (0,0)
currentRedVelocity = 0

blueAndSnare = False
blueAndHiHat = False
redAndSnare = False
redAndHiHat = False
booli  = [False for i in range(2)]

numDrums = 0
drums = [None for i in range(2)]
def newDrum(pos, name):
    # pos = (x, y)
    drum = cv2.circle(img,pos, 50,(0,0, 0),5)
    cv2.putText(drum,name,pos,cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    blank = np.zeros(img.shape[0:2])
    drum_image = cv2.circle(blank.copy(), pos, 50, (255, 255, 255), -5)
    global numDrums
    numDrums += 1
    return (name, drum_image)


while(1):
    now = time.time()
    fps = frameCount / (now - timeStart)
    frameCount += 1

    _, img = cap.read()
    img = cv2.flip(img, 1)

    # cv2.putText(img,"FPS : ",(10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(img,"FPS: %.2f" % (fps),(10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    # Add the drums
    drums[0] = newDrum((350, 400), "snare")
    drums[1] = newDrum((100, 400), "hi_hat")

    #converting frame(img i.e BGR) to HSV (hue-saturation-value)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #defining the range of red color
    red_lower=np.array([255,255,255],np.uint8)
    red_upper=np.array([255,255,255],np.uint8)

    #defining the Range of Blue color
    blue_lower=np.array([95,60,94],np.uint8)
    blue_upper=np.array([163,168,209],np.uint8)

    #finding the range of red,blue color in the image
    red=cv2.inRange(hsv, red_lower, red_upper)
    blue=cv2.inRange(hsv,blue_lower,blue_upper)

    #Morphological transformation, Dilation
    kernal = np.ones((5 ,5), "uint8")

    red=cv2.dilate(red, kernal)
    res=cv2.bitwise_and(img, img, mask = red)

    blue=cv2.dilate(blue,kernal)
    res1=cv2.bitwise_and(img, img, mask = blue)


    #Tracking the Red Color
    (_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    (redCenter, redEllipse) = drawEllipse(contours, "Red")
    # cv2.drawContours(img, contours, -1 , (0,0,255), 2)


    #Tracking the Blue Color
    (_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1 , (255,0,0), 2)
    (blueCenter, blueEllipse) = drawEllipse(contours, "Blue")

    b1 = b2
    b2 = blueCenter
    bDelta = math.sqrt((b2[0] - b1[0])**2 + (b2[1] - b1[1])**2)
    bVelocity = bDelta * fps / 100
    if (bVelocity - currentBlueVelocity) > 10:
        cv2.putText(img,str(int(bVelocity)),(10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    else:
        cv2.putText(img,str(int(currentBlueVelocity)),(10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    currentBlueVelocity = bVelocity

    r1 = r2
    r2 = redCenter
    rDelta = math.sqrt((r2[0] - r1[0])**2 + (r2[1] - r1[1])**2)
    rVelocity = rDelta * fps / 100
    if (rVelocity - currentRedVelocity) > 10:
        cv2.putText(img,str(int(rVelocity)),(70, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        cv2.putText(img,str(int(currentRedVelocity)),(70, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    currentRedVelocity = rVelocity


    for i in range(len(drums)):
        # print(booli)
        booli[i] = detectCollision(blueEllipse, drums[i][1], currentBlueVelocity, booli[i], "{0}.wav".format(drums[i][0]))
    # blueAndSnare = detectCollision(blueEllipse, drums[0][1], blueAndSnare, "snare.wav")
    # blueAndHiHat = detectCollision(blueEllipse, drums[1][1], blueAndHiHat, "hi_hat.wav")

    # blueAndSnare = detectCollision(blueEllipse, snare_image, blueAndSnare, "snare.wav")
    # blueAndHiHat = detectCollision(blueEllipse, hi_hat_image, blueAndHiHat, "Closed-Hi-Hat.wav")
    #
    # redAndSnare = detectCollision(redEllipse, snare_image, redAndSnare, "snare.wav")
    # redAndHiHat = detectCollision(redEllipse, hi_hat_image, redAndHiHat, "Closed-Hi-Hat.wav")



    #cv2.imshow("Redcolour",red)
    cv2.imshow("Color Tracking",img)
    #cv2.imshow("red",res)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
